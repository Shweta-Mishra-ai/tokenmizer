"""
Authentication middleware.

When TOKENMIZER_API_KEY is set, all non-health endpoints require:
  Authorization: Bearer <key>
  or
  X-API-Key: <key>

When empty (default), auth is skipped — dev mode.
Uses constant-time comparison to prevent timing attacks.
"""
from __future__ import annotations
import hashlib
import hmac
from fastapi import Request, HTTPException


def _get_configured_key() -> str:
    """Read from settings — single source of truth, not scattered os.getenv calls."""
    try:
        from tokenmizer.config.settings import get_settings
        return get_settings().api_key
    except Exception:
        return ""


async def verify_api_key(request: Request) -> None:
    """FastAPI dependency. Raises 401 if auth fails."""
    configured = _get_configured_key()
    if not configured:
        return  # dev mode — auth disabled

    # Try Authorization header first, then X-API-Key
    key = ""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        key = auth_header.removeprefix("Bearer ").strip()
    if not key:
        key = request.headers.get("X-API-Key", "").strip()

    if not key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide Authorization: Bearer <key> or X-API-Key: <key>",
        )

    # Constant-time comparison — prevents timing side-channel attacks
    valid = hmac.compare_digest(
        hashlib.sha256(key.encode()).digest(),
        hashlib.sha256(configured.encode()).digest(),
    )
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid API key")
