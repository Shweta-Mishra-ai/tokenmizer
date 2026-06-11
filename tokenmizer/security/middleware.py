"""Prompt injection detection middleware."""
from __future__ import annotations

import logging
import re

# FastAPI imported lazily — _scan_messages works without it

logger = logging.getLogger(__name__)

_INJECTION_PATTERNS = [
    re.compile(r'ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?', re.IGNORECASE),
    re.compile(r'disregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?', re.IGNORECASE),
    re.compile(r'you\s+are\s+now\s+(?:DAN|jailbreak|unrestricted)', re.IGNORECASE),
    re.compile(r'print\s+(?:your\s+)?(?:system\s+prompt|instructions)', re.IGNORECASE),
    re.compile(r'reveal\s+(?:your\s+)?(?:system\s+prompt|instructions|api\s+key)', re.IGNORECASE),
    re.compile(r'bypass\s+(?:your\s+)?(?:safety|filter|restriction)', re.IGNORECASE),
]


def _scan_messages(messages: list[dict]) -> bool:
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        for pat in _INJECTION_PATTERNS:
            if pat.search(content):
                return True
    return False


async def injection_guard(request) -> None:
    """FastAPI dependency. Raises 429 on detected injection attempt."""
    try:
        from fastapi import HTTPException
    except ImportError:
        return  # not running under FastAPI — skip
    if request.method == "POST" and "chat" in request.url.path:
        try:
            body = await request.json()
            messages = body.get("messages", [])
            if _scan_messages(messages):
                logger.warning(f"Injection attempt blocked from {request.client.host}")
                raise HTTPException(
                    status_code=429,
                    detail="Request blocked: prompt injection detected",
                )
        except HTTPException:
            raise
        except Exception:
            pass  # don't block on parse errors
