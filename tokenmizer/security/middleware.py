"""
Basic prompt-injection KEYWORD FILTER.

HONESTY NOTE (this used to be called "prompt injection detection" — that
name overclaimed what regex matching can do, and the project README/
SECURITY.md repeated the overclaim as a security guarantee):

What this module actually is: a denylist of ~10 common, unsophisticated
injection phrasings ("ignore previous instructions", "reveal your system
prompt", etc). It will catch copy-pasted jailbreak templates found on the
open web. It will NOT catch:
  - paraphrased or reworded injection attempts
  - injection in any language other than English
  - base64/rot13/unicode-homoglyph encoded payloads
  - injection split across multiple messages or hidden in tool results
  - novel phrasings not in this list

This is a best-effort speed bump against the laziest attempts, not a
security boundary. Do not present it as "prompt injection detection" or
"protection" in customer-facing docs without this caveat attached —
a security-literate reviewer will (correctly) discount the whole
product's security claims the moment they read this file and see a
10-pattern regex list being called "detection."

If you need real injection defense: structurally separate untrusted
content from instructions (e.g. tool-result fencing), keep the LLM's
privilege scope minimal regardless of what's in its context, and treat
this filter as one weak, optional layer among several — never the only one.
"""
from __future__ import annotations

import logging
import re

# Request must be importable at module scope: injection_guard is a FastAPI
# dependency and its `request` parameter MUST carry a resolvable `Request`
# annotation. Without it, FastAPI treats `request` as a required string
# QUERY parameter and every POST /v1/chat/completions gets a 422
# ("query.request missing") — the endpoint is dead for all clients.
# _scan_messages still works without fastapi installed (fallback below).
try:
    from fastapi import Request
except ImportError:  # pragma: no cover — non-FastAPI usage of _scan_messages
    Request = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

_INJECTION_PATTERNS = [
    re.compile(r'ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?', re.IGNORECASE),
    re.compile(r'disregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?', re.IGNORECASE),
    re.compile(r'forget\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?', re.IGNORECASE),
    re.compile(r'you\s+are\s+now\s+(?:DAN|jailbreak(?:ed)?|unrestricted|in\s+developer\s+mode)', re.IGNORECASE),
    re.compile(r'print\s+(?:your\s+)?(?:system\s+prompt|instructions)', re.IGNORECASE),
    re.compile(r'reveal\s+(?:your\s+)?(?:system\s+prompt|instructions|api\s+key)', re.IGNORECASE),
    re.compile(r'repeat\s+(?:the\s+|your\s+)?(?:words?\s+)?above', re.IGNORECASE),
    re.compile(r'bypass\s+(?:your\s+)?(?:safety|filter|restriction|guardrails?)', re.IGNORECASE),
    re.compile(r'pretend\s+(?:you\s+have\s+no|there\s+(?:are|is)\s+no)\s+(?:restrictions?|rules?|filters?)', re.IGNORECASE),
    re.compile(r'this\s+is\s+a\s+(?:test|simulation)[,.]?\s+(?:ignore|disregard|skip)', re.IGNORECASE),
]


def _content_to_plain_text(content) -> str:
    """Flatten any message-content shape (str / list-of-blocks / dict / None)
    to plain text for scanning. Mirrors graph_memory.helpers._content_to_text
    but is duplicated here (not imported) to keep the security module
    dependency-free of graph internals — this should be safe to use even
    if graph_memory is broken or removed."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]
    return ""


def _scan_messages(messages: list[dict]) -> bool:
    """Returns True if any message matches a known unsophisticated-injection
    pattern. See module docstring for what this does NOT catch.

    FIXED: previously skipped any non-str content entirely (`if not
    isinstance(content, str): continue`), meaning multimodal messages —
    text + image, or text wrapped in content blocks — were never scanned
    at all, silently. An attacker only had to wrap injection text in a
    one-element content-block list to bypass the filter completely."""
    for msg in messages:
        text = _content_to_plain_text(msg.get("content"))
        if not text:
            continue
        for pat in _INJECTION_PATTERNS:
            if pat.search(text):
                return True
    return False


async def injection_guard(request: Request) -> None:
    """FastAPI dependency. Raises 400 on a matched denylist phrase.

    FIXED: previously raised 429 (Too Many Requests), which is semantically
    wrong and actively misleading to API consumers — 429 tells a well-behaved
    client "retry with backoff," but retrying an injection-flagged request
    will fail identically every time. 400 (Bad Request) is correct: the
    request itself was rejected, not rate-limited."""
    try:
        from fastapi import HTTPException
    except ImportError:
        return  # not running under FastAPI — skip

    if request.method == "POST" and "chat" in request.url.path:
        try:
            body = await request.json()
            messages = body.get("messages", [])
            if _scan_messages(messages):
                client_host = request.client.host if request.client else "unknown"
                logger.warning(f"Injection-pattern match blocked from {client_host}")
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Request blocked: matched a known prompt-injection phrasing. "
                        "This is a basic keyword filter, not comprehensive protection."
                    ),
                )
        except HTTPException:
            raise
        except Exception as e:
            # Non-blocking: malformed body or parse error shouldn't deny the request.
            # Logged at warning (not debug) — a parse failure here means the guard
            # silently let a request through unscanned, which is worth knowing about
            # even though we don't want to block on it.
            logger.warning(f"Injection scan skipped (parse error, request allowed through): {e}")
