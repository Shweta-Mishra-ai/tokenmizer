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
import re
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

from tokenmizer.core.tokenizer import count_messages_tokens, count_tokens
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
        tool_index_tokens: int = 0,
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
        # Budget, in tokens, for the index of paths seen in the tool output
        # that windowing drops. See path_index. 0 leaves it out.
        self.tool_index_tokens = tool_index_tokens
        # session id -> (messages windowed out, hash of them, bridge message,
        # conversation-level lead-in). Bounded: a proxy serves many sessions
        # and this is only a cost optimisation — a forgotten entry re-cuts,
        # it never loses a turn.
        self._frozen: "OrderedDict[str, tuple[int, str, dict, list]]" = OrderedDict()
        self._frozen_max = 10_000
        # apply() runs off the event loop, on whichever worker thread the
        # request landed on, and two sessions' requests can be in it at once.
        # The table is shared between them; the conversation and the graph
        # are not (a session's own lock serialises those).
        self._frozen_lock = threading.Lock()

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

        # What the agent looked at in the tool output being dropped. The
        # graph holds the files it knows are important, in a few dozen
        # tokens; the agent goes on to use files and directories that a
        # listing or a search showed it once and no node records.
        if self.tool_index_tokens > 0:
            index = path_index(old, self.tool_index_tokens, model)
            if index:
                bridge_parts.append(fence(index, "paths seen in earlier tool output"))

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
        with self._frozen_lock:
            frozen = self._frozen.get(session_id)
        if frozen is None:
            return None
        split, digest, bridge_msg, lead = frozen
        if split >= len(conv_msgs) or _digest(conv_msgs[:split]) != digest:
            return None
        candidate = system_msgs + [bridge_msg] + [dict(m) for m in lead] + conv_msgs[split:]
        # See _tail_limit: the chat budget in chat, the ceiling in an agent
        # loop.
        if count_messages_tokens(candidate, model) > self._tail_limit(conv_msgs):
            return None
        with self._frozen_lock:
            if session_id in self._frozen:
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
        if self.max_tail_tokens > 0 and any(m.get("role") == "tool" for m in conv):
            # In an agent loop, cut down to half the ceiling, not to just
            # under it and not merely to protect_recent messages. A tail
            # left near the ceiling has no room to grow, so the next step
            # pushes it over and the cut is made again, moving the start of
            # the request on every request and never letting the provider's
            # cache serve the history. Cut to half, the tail grows back over
            # several requests before the next cut.
            target = self.max_tail_tokens // 2
            while split < last and count_messages_tokens(conv[split:], model) > target:
                split = next(i for i in starts if i > split)
        return split

    def _tail_limit(self, conv: list[dict]) -> int:
        """How large a held cut may grow before it is re-made.

        In chat that is the token budget, as it always was. In an agent loop
        (there are tool results in the conversation) a tail is dominated by
        tool output and never fits a chat-sized budget, so the ceiling is the
        limit; holding to the budget would re-cut on every request.
        """
        if self.max_tail_tokens > 0 and any(m.get("role") == "tool" for m in conv):
            return self.max_tail_tokens
        return self.token_budget

    def _freeze(self, session_id: str, split: int, digest: str,
                bridge_msg: dict, lead: list) -> None:
        with self._frozen_lock:
            self._frozen[session_id] = (split, digest, bridge_msg, lead)
            self._frozen.move_to_end(session_id)
            while len(self._frozen) > self._frozen_max:
                self._frozen.popitem(last=False)


_PATH_EXTENSIONS = frozenset((
    "py js jsx ts tsx json md toml yaml yml txt sh cfg ini html css go rs java "
    "c h cpp rb sql lock xml csv env ipynb").split())
_PATH_STRIP = "\"'`()[]{}<>,;:*|=!?"
_MAX_TOKEN_CHARS = 160          # longer is a blob, not a path
_MAX_SCAN_CHARS = 400_000       # per message; the rest is not read
_MAX_INDEX_CHARS = 2_000_000    # per index, newest output first
_PATH_CHARS = re.compile(r"[\w.\-/@+~]+")


def _as_path(word: str) -> str:
    """`word` as a path, or "" if it is not one.

    Word by word rather than one regex over the whole text: tool output has
    minified lines and blobs with no separator for hundreds of thousands of
    characters, and a pattern with nested repeats over them is quadratic.
    Here every word is looked at once, and long ones are skipped unread.
    """
    word = word.strip(_PATH_STRIP).rstrip(".")
    if not (4 <= len(word) <= _MAX_TOKEN_CHARS) or "://" in word or word[0] == "-":
        return ""
    if not _PATH_CHARS.fullmatch(word):
        return ""
    word = word.removeprefix("./")
    if "/" in word:
        # a directory, or a file in one; "a/b" is not a fraction or a date
        # when it has a letter in it and no more than one dot-free segment
        # of digits alone
        return word if any(c.isalpha() for c in word) else ""
    stem, dot, ext = word.rpartition(".")
    return word if dot and stem and ext.lower() in _PATH_EXTENSIONS else ""


def path_index(dropped: list[dict], budget: int, model: str = "gpt-4o") -> str:
    """The paths mentioned in the tool calls and tool output that windowing
    is dropping, grouped by directory, newest directories first, within
    `budget` tokens. "" when there are none.

    Measured on a real agent session, 60% of what the agent went on to use
    and no longer had were file paths, and another third were parts of
    them; they were in a listing or a search result that had been dropped.
    No selector predicts which of a request's ~280 paths will be needed
    (the needed path's rank by recency or frequency was about 176 of 276),
    so the index lists them all and lets the budget decide. Written the way
    a directory listing writes them, which is also about 40% smaller than
    full paths.
    """
    seen: dict[str, None] = {}
    # A cut runs on the request path and reads everything it drops, so the
    # work has to be bounded however long the session is: read the newest
    # output back to a total cap and leave the oldest unread. A very long
    # session loses paths that only its earliest output showed, which is the
    # right end to lose them from.
    budget_chars = _MAX_INDEX_CHARS
    recent: list[dict] = []
    for m in reversed(dropped):
        size = len(m.get("content") or "") if isinstance(m.get("content"), str) else 0
        for call in m.get("tool_calls") or []:
            size += len((call.get("function") or {}).get("arguments") or "")
        if recent and budget_chars - size < 0:
            break
        budget_chars -= size
        recent.append(m)
    for m in reversed(recent):
        texts = []
        content = m.get("content")
        # Tool output and the calls that made it, not what people wrote: a
        # chat that mentions a file is not an agent that has looked at it.
        if m.get("role") == "tool" and isinstance(content, str):
            texts.append(content)
        for call in m.get("tool_calls") or []:
            texts.append((call.get("function") or {}).get("arguments") or "")
        for text in texts:
            for word in text[:_MAX_SCAN_CHARS].split():
                path = _as_path(word)
                if path:
                    seen.setdefault(path)
    if not seen:
        return ""
    groups: dict[str, list[str]] = {}
    for path in seen:
        directory, _, name = path.rpartition("/")
        groups.setdefault(directory + "/" if directory else "", []).append(name)
    head = ("Paths seen in earlier tool output (contents omitted; read a file "
            "again to see it):")
    lines = [f"{d or '(no directory)'}: {' '.join(names)}" for d, names in groups.items()]
    used = count_tokens(head, model)
    kept: list[str] = []
    for line in reversed(lines):              # newest directories win
        cost = count_tokens(line, model) + 1
        if used + cost > budget:
            break
        kept.append(line)
        used += cost
    if not kept:
        return ""
    return head + "\n" + "\n".join(reversed(kept))


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
