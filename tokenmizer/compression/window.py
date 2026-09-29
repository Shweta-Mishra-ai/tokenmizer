"""
Smart Message Window — kills the biggest token drain in long sessions.

Problem:
  In a 50-turn session, turns 1-40 are sent verbatim EVERY single turn.
  50 turns × avg 300 tokens/turn = 15,000 tokens repeated each time.
  At Opus 4.8 pricing ($5/M): 15,000 × 50 turns = 750K tokens = $3.75
  just in conversation history repetition.

Solution:
  Keep the last N turns verbatim (recent context).
  Replace older turns with the graph memory context block.
  The graph has the important information — tasks, decisions, files.
  The LLM doesn't need the full conversation text to know what was done.

Quality guarantee:
  - System messages always preserved
  - Last N turns always verbatim (configurable, default 8)
  - Graph context is accurate (SQLite-backed, not ephemeral)
  - No hallucination risk: graph only contains extracted facts
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

from tokenmizer.core.tokenizer import count_messages_tokens
from tokenmizer.security.fencing import fence

if TYPE_CHECKING:
    from tokenmizer.graph_memory.graph import GraphMemory

logger = logging.getLogger(__name__)


class SmartMessageWindow:

    def __init__(
        self,
        token_budget: int = 4000,
        protect_recent: int = 8,
        graph_context_budget: int = 250,
        max_tail_tokens: int = 0,
    ):
        self.token_budget = token_budget
        self.protect_recent = protect_recent
        self.graph_context_budget = graph_context_budget
        # Ceiling on the verbatim tail, in tokens. protect_recent counts
        # messages, and in an agent loop one message is a file or a test log,
        # so ten of them can be most of the payload. Add to that a window
        # that used to open on a user turn, of which an agent loop has
        # almost none, and the "protected tail" was nearly the whole
        # conversation and windowing removed almost nothing.
        # Whole steps are dropped from the front of the tail until it fits,
        # but the newest step is never dropped. 0 disables the ceiling.
        self.max_tail_tokens = max_tail_tokens
        # session id -> (messages windowed out, hash of them, bridge message,
        # conversation-level lead-in). Bounded: a proxy serves many sessions
        # and this is only a cost optimisation — a forgotten entry re-cuts,
        # it never loses a turn.
        self._frozen: "OrderedDict[str, tuple[int, str, dict, list]]" = OrderedDict()
        self._frozen_max = 10_000

    def apply(
        self,
        messages: list[dict],
        graph: "GraphMemory",
        model: str = "gpt-4o",
        stable: bool = False,
    ) -> tuple[list[dict], int]:
        """
        Apply smart windowing to messages.

        `stable` keeps the cut where it was until the verbatim tail outgrows
        the budget again, instead of sliding it forward one turn per
        request. See _reuse_frozen_cut.

        Returns:
            (windowed_messages, tokens_saved)
        """
        current_tokens = count_messages_tokens(messages, model)

        if current_tokens <= self.token_budget:
            return messages, 0  # fits — don't touch

        system_msgs = [m for m in messages if m.get("role") == "system"]
        conv_msgs = [m for m in messages if m.get("role") != "system"]

        session_id = getattr(graph, "session_id", "") or ""
        if stable and session_id:
            reused = self._reuse_frozen_cut(
                session_id, system_msgs, conv_msgs, model)
            if reused is not None:
                return reused, max(0, current_tokens - count_messages_tokens(reused, model))

        if len(conv_msgs) <= self.protect_recent:
            return messages, 0  # not enough history to window

        split = self._choose_split(conv_msgs, model)

        if split is None:
            # No step to open the window on — nothing safe to window down
            # to. Send the conversation unchanged rather than an empty one;
            # the provider guard would otherwise reject it.
            return messages, 0

        recent = conv_msgs[split:]
        old = conv_msgs[:split]
        # Both providers want a conversation that opens on a user turn. When
        # the window opens on an agent step instead, say why the first turn
        # is an assistant's, in words that never change so the request
        # prefix stays byte-identical from one request to the next.
        lead = [] if recent[0].get("role") == "user" else [dict(_LEAD_IN)]

        # Before the old turns are replaced, keep what the ontology has no
        # node for — a stated constraint, a budget, a deadline. Those are
        # the facts that leave the session permanently at this point, and
        # nothing else in the pipeline is looking for them. Best-effort by
        # design: a failure here must not cost the caller their answer, and
        # the windowing below is correct without it.
        try:
            graph.record_span_summary(old)
        except Exception as e:                      # pragma: no cover - defensive
            logger.warning(
                "Span summary failed for %s (windowing continues, but the "
                "constraints stated in the dropped turns are now lost): %s",
                getattr(graph, "session_id", "?"), e,
            )

        # Build graph context to replace old turns
        graph_ctx = graph.to_context_block(token_budget=self.graph_context_budget)

        bridge_parts = []
        if graph_ctx:
            # Fenced for the same reason the proxy fences its context block:
            # this is conversation text being promoted into a system
            # message, where an imperative reads as an instruction rather
            # than as a record of one. See security/fencing.py.
            bridge_parts.append(fence(graph_ctx, "session context from earlier turns"))

        # Add a note about what's omitted
        bridge_parts.append(
            f"[{len(old)} earlier messages omitted — key information preserved above]"
        )

        bridge_msg = {
            "role": "system",
            "content": "\n\n".join(bridge_parts),
        }

        windowed = system_msgs + [bridge_msg] + lead + recent
        windowed_tokens = count_messages_tokens(windowed, model)
        saved = current_tokens - windowed_tokens

        if stable and session_id:
            self._freeze(session_id, split, _digest(old), bridge_msg, lead)

        logger.info(
            f"SmartWindow: {len(old)} old turns compressed → "
            f"{current_tokens}→{windowed_tokens} tokens (saved {saved})"
        )

        return windowed, max(0, saved)


    def _reuse_frozen_cut(
        self,
        session_id: str,
        system_msgs: list[dict],
        conv_msgs: list[dict],
        model: str,
    ) -> "list[dict] | None":
        """The previous request's cut and bridge, if they still apply.

        Sliding the window forward one turn per request replaces the bridge
        and the first verbatim turn every time, so no two requests share a
        prefix and a provider's prompt cache never serves one. Holding the
        cut still makes every request between two cuts the previous one
        plus a turn: the system prompt, the bridge and the history all go
        out byte-identical and are billed at the cache-read price.

        The cut is reused only while the turns it replaced are exactly the
        ones this request starts with (the client may edit or regenerate
        history) and the verbatim tail still fits the budget. Otherwise
        the caller cuts afresh. Nothing is ever dropped that the sliding
        window would have kept: turns after the cut stay verbatim.
        """
        frozen = self._frozen.get(session_id)
        if frozen is None:
            return None
        split, digest, bridge_msg, lead = frozen
        if split >= len(conv_msgs) or _digest(conv_msgs[:split]) != digest:
            return None
        candidate = system_msgs + [bridge_msg] + [dict(m) for m in lead] + conv_msgs[split:]
        # The budget is for chat, where a turn is a few hundred tokens. In
        # an agent loop the tail is dominated by tool output and never fits
        # it, so holding the cut to the budget would re-cut on every request
        # and hand back nothing. The tail ceiling is the limit there.
        if count_messages_tokens(candidate, model) > max(self.token_budget,
                                                         self.max_tail_tokens):
            return None
        self._frozen.move_to_end(session_id)
        return candidate

    def _choose_split(self, conv: list[dict], model: str) -> "int | None":
        """Where the verbatim tail begins, or None if there is no safe place.

        A tail may only open at the start of a step: a user turn, or an
        assistant turn that calls tools, which brings its tool results with
        it. Opening on a tool result strands it from the call it answers,
        and both providers reject that.

        Starts from the last `protect_recent` messages, moves forward to the
        next step, then, if a ceiling is set and the tail is over it, drops
        whole steps from the front until it fits. The newest step always
        stays: it is the one the model is about to act on.
        """
        starts = [i for i, m in enumerate(conv)
                  if m.get("role") == "user"
                  or (m.get("role") == "assistant" and m.get("tool_calls"))]
        if not starts or len(conv) <= self.protect_recent:
            return None
        last = starts[-1]
        wanted = len(conv) - self.protect_recent
        split = next((i for i in starts if i >= wanted), last)
        if self.max_tail_tokens > 0:
            while split < last and count_messages_tokens(
                    conv[split:], model) > self.max_tail_tokens:
                split = next(i for i in starts if i > split)
        return split

    def _freeze(self, session_id: str, split: int, digest: str,
                bridge_msg: dict, lead: list) -> None:
        self._frozen[session_id] = (split, digest, bridge_msg, lead)
        self._frozen.move_to_end(session_id)
        while len(self._frozen) > self._frozen_max:
            self._frozen.popitem(last=False)


# What opens the tail when the window starts on an agent step. Constant, so
# it never disturbs the byte-identical prefix a provider's cache depends on.
_LEAD_IN = {
    "role": "user",
    "content": "[The earlier steps of this session are summarised in the "
               "session context above. Continue from the steps below.]",
}


def _digest(messages: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(messages, sort_keys=True, default=str).encode()
    ).hexdigest()


def needs_windowing(messages: list[dict], token_budget: int, model: str = "gpt-4o") -> bool:
    """Quick check — should we apply windowing?"""
    return count_messages_tokens(messages, model) > token_budget
