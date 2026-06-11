"""
Accurate token counting using tiktoken.
Replaces every `len(text) // 4` in the codebase.
"""
from __future__ import annotations

import functools

_FALLBACK_RATIO = 4  # chars per token — only used if tiktoken unavailable


@functools.lru_cache(maxsize=16)
def _get_encoding(model: str):
    try:
        import tiktoken
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        return None


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Accurate token count. Falls back to char/4 if tiktoken not installed."""
    if not text:
        return 0
    enc = _get_encoding(model)
    if enc is None:
        return max(1, len(text) // _FALLBACK_RATIO)
    return len(enc.encode(text, disallowed_special=()))


def count_messages_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
    """Count tokens across all messages including OpenAI/Anthropic role overhead."""
    total = 0
    for msg in messages:
        total += 4  # per-message framing tokens
        total += count_tokens(msg.get("content", ""), model)
        total += count_tokens(msg.get("role", ""), model)
    total += 2  # reply priming
    return total


def chars_to_tokens_estimate(chars: int) -> int:
    """Fast estimate when we only have char count (e.g. for size checks)."""
    return max(1, chars // _FALLBACK_RATIO)
