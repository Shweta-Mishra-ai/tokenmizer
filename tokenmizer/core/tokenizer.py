"""
Accurate token counting.

- OpenAI/compatible models: tiktoken (model-specific encoding or cl100k_base fallback)
- Anthropic/Claude models: tiktoken is the WRONG tokenizer — Claude uses a different
  vocabulary and previously every Claude request was counted with an OpenAI encoder,
  which is inaccurate (typically 5-20% off, worse on code-heavy content). This module
  now routes Claude models through the Anthropic SDK's local tokenizer when available,
  and only falls back to the tiktoken approximation if the SDK doesn't expose one.
"""
from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)

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


def is_claude_model(model: str) -> bool:
    """True if this is an Anthropic/Claude model — needs the Anthropic tokenizer,
    not tiktoken."""
    return "claude" in model.lower() or "anthropic" in model.lower()


def _count_with_anthropic_sdk(text: str) -> int | None:
    """
    Try the Anthropic SDK's local tokenizer.
    Older SDK (<0.20): anthropic.count_tokens(text)
    Some intermediate versions: anthropic.tokenizer.count_tokens(text)
    Newer SDK versions removed the local tokenizer entirely (requires an API call) —
    in that case this returns None and the caller falls back to a tiktoken estimate.
    """
    try:
        import anthropic as _anthropic
        if hasattr(_anthropic, "count_tokens"):
            return int(_anthropic.count_tokens(text))
        if hasattr(_anthropic, "tokenizer") and hasattr(_anthropic.tokenizer, "count_tokens"):
            return int(_anthropic.tokenizer.count_tokens(text))
    except Exception as e:
        # FIXED: previously a bare `except Exception: pass` — the
        # DOCUMENTED case (SDK installed but no local count_tokens
        # exposed, so we fall back to the tiktoken approximation) is
        # fine to stay quiet about, but an UNEXPECTED failure (the SDK
        # has count_tokens and it raises for some other reason) was
        # silently degrading every Claude-model token count with zero
        # visibility. Logged at debug — this runs on every request, so
        # anything louder would be noisy — but no longer invisible to a
        # maintainer investigating token-count drift.
        logger.debug(
            f"Anthropic SDK count_tokens call failed, falling back to "
            f"tiktoken approximation: {type(e).__name__}: {e}"
        )
    return None


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Accurate token count for the given model.

    Claude/Anthropic models: tries the Anthropic SDK's local tokenizer first, falls
    back to a cl100k_base (tiktoken) approximation if the SDK doesn't expose one —
    this approximation carries a documented error margin, see module docstring.
    All other models: tiktoken with the correct model-specific encoding.
    Falls back to char/4 if tiktoken is not installed at all.
    """
    if not text:
        return 0

    if is_claude_model(model):
        sdk_count = _count_with_anthropic_sdk(text)
        if sdk_count is not None:
            return sdk_count
        enc = _get_encoding("gpt-4o")  # closest available approximation
        if enc is not None:
            return len(enc.encode(text, disallowed_special=()))
        return max(1, len(text) // _FALLBACK_RATIO)

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
