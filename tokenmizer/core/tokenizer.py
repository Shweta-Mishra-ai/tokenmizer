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
    """
    Resolve a tiktoken encoding, or None if one cannot be obtained for
    ANY reason. Never raises.

    This used to catch only ImportError. tiktoken does not ship its BPE
    vocabulary — `encoding_for_model()` downloads it from
    openaipublic.blob.core.windows.net on first use — so on an
    air-gapped host, behind an egress proxy, or during a blob outage it
    raises a network error (requests.ProxyError / ConnectionError),
    which sailed straight past that ImportError handler and out through
    count_tokens(). count_messages_tokens() is on the hot path of every
    proxied request, so a transient failure to reach a third-party CDN
    turned into a 500 on EVERY request, and the documented char/4
    fallback below was unreachable — it only ever ran when tiktoken was
    not installed at all.

    Catching broadly here is deliberate: an approximate token count is
    always better than a dead proxy. The result (including None) is
    cached by lru_cache, so a failure costs one attempt per model rather
    than a fresh network timeout on every single request.

    To avoid the network entirely, pre-fetch the vocabulary at image
    build time and point TIKTOKEN_CACHE_DIR at it — see the Dockerfile.
    """
    try:
        import tiktoken
    except ImportError:
        logger.warning(
            "tiktoken is not installed — falling back to a char/%d token "
            "estimate. Install tiktoken for accurate counts.", _FALLBACK_RATIO
        )
        return None

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        pass  # unknown model name — fall through to the generic encoding
    except Exception as e:
        logger.warning(
            "tiktoken could not load an encoding for model %r (%s: %s). "
            "Falling back to a char/%d estimate for this model. If this is "
            "a network error, the BPE vocabulary could not be downloaded — "
            "pre-populate TIKTOKEN_CACHE_DIR to run without egress.",
            model, type(e).__name__, e, _FALLBACK_RATIO,
        )
        return None

    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logger.warning(
            "tiktoken could not load the cl100k_base fallback encoding "
            "(%s: %s). Using a char/%d estimate — token counts will be "
            "approximate until this is resolved.",
            type(e).__name__, e, _FALLBACK_RATIO,
        )
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
        return _encode_len(_get_encoding("gpt-4o"), text)  # closest approximation

    return _encode_len(_get_encoding(model), text)


def _encode_len(enc, text: str) -> int:
    """Token count via `enc`, or the char/N estimate if `enc` is None or
    the encode call itself fails. Counting tokens must never be able to
    fail a request — the count is an optimisation input, not an answer
    the caller asked for."""
    if enc is None:
        return max(1, len(text) // _FALLBACK_RATIO)
    try:
        return len(enc.encode(text, disallowed_special=()))
    except Exception as e:
        logger.warning(
            "Token encode failed (%s: %s) — using a char/%d estimate for "
            "this call.", type(e).__name__, e, _FALLBACK_RATIO,
        )
        return max(1, len(text) // _FALLBACK_RATIO)


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
