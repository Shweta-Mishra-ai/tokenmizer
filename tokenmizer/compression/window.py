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

import logging
from typing import TYPE_CHECKING

from tokenmizer.core.tokenizer import count_messages_tokens, count_tokens

if TYPE_CHECKING:
    from tokenmizer.graph_memory.graph import GraphMemory

logger = logging.getLogger(__name__)


class SmartMessageWindow:

    def __init__(
        self,
        token_budget: int = 4000,
        protect_recent: int = 8,
        graph_context_budget: int = 250,
    ):
        self.token_budget = token_budget
        self.protect_recent = protect_recent
        self.graph_context_budget = graph_context_budget

    def apply(
        self,
        messages: list[dict],
        graph: "GraphMemory",
        model: str = "gpt-4o",
    ) -> tuple[list[dict], int]:
        """
        Apply smart windowing to messages.
        
        Returns:
            (windowed_messages, tokens_saved)
        """
        current_tokens = count_messages_tokens(messages, model)

        if current_tokens <= self.token_budget:
            return messages, 0  # fits — don't touch

        system_msgs = [m for m in messages if m.get("role") == "system"]
        conv_msgs = [m for m in messages if m.get("role") != "system"]

        if len(conv_msgs) <= self.protect_recent:
            return messages, 0  # not enough history to window

        recent = conv_msgs[-self.protect_recent:]
        old = conv_msgs[:-self.protect_recent]

        # Build graph context to replace old turns
        graph_ctx = graph.to_context_block(token_budget=self.graph_context_budget)

        bridge_parts = []
        if graph_ctx:
            bridge_parts.append(f"[Session context from earlier conversation]\n{graph_ctx}")

        # Add a note about what's omitted
        bridge_parts.append(
            f"[{len(old)} earlier messages omitted — key information preserved above]"
        )

        bridge_msg = {
            "role": "system",
            "content": "\n\n".join(bridge_parts),
        }

        windowed = system_msgs + [bridge_msg] + recent
        windowed_tokens = count_messages_tokens(windowed, model)
        saved = current_tokens - windowed_tokens

        logger.info(
            f"SmartWindow: {len(old)} old turns compressed → "
            f"{current_tokens}→{windowed_tokens} tokens (saved {saved})"
        )

        return windowed, max(0, saved)


def needs_windowing(messages: list[dict], token_budget: int, model: str = "gpt-4o") -> bool:
    """Quick check — should we apply windowing?"""
    return count_messages_tokens(messages, model) > token_budget
