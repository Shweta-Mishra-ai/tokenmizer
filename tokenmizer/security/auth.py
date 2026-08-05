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
import logging

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)


class _ConfigReadError(Exception):
    """Raised internally when settings can't be read — distinct from
    'no key configured' so verify_api_key can fail closed instead of
    silently treating a config error as 'auth disabled.'"""


def _get_configured_keys() -> list[str]:
    """All credentials that grant access, primary key first.

    `api_key` stays the primary (single-tenant) credential. `api_keys`
    adds further ones, each of which becomes its OWN principal for
    session-ownership purposes — that's what makes genuine isolation
    between callers possible (see security/ownership.py). With zero or
    one key configured, nothing about the deployment changes.
    """
    from tokenmizer.config.settings import get_settings
    settings = get_settings()
    keys = [settings.api_key] if settings.api_key else []
    keys.extend(k for k in getattr(settings, "api_keys", []) if k)
    return keys


def _get_configured_key() -> str:
    """Read from settings — single source of truth, not scattered os.getenv calls.

    FIXED — CRITICAL SECURITY BUG: this previously caught ANY exception
    from get_settings() and returned "" unconditionally. verify_api_key()
    treats an empty configured key as "dev mode — auth disabled" and lets
    the request through with NO authentication check at all. That means
    any transient error in settings loading (a bug, a race during startup,
    a future refactor that adds a fallible step to get_settings()) would
    silently and completely disable authentication on every endpoint,
    fail-OPEN, with no log line, no alert, nothing — the worst possible
    failure mode for an auth check. Authentication must fail CLOSED: if we
    can't determine whether a key is configured, we must treat that as
    "yes, a key is required" and reject the request, not wave it through.
    """
    try:
        from tokenmizer.config.settings import get_settings
        return get_settings().api_key
    except Exception as e:
        logger.error(
            f"Failed to read configured API key from settings: {e}. "
            "Failing CLOSED (rejecting requests) rather than disabling "
            "auth — see security/auth.py for why this must never fail open."
        )
        raise _ConfigReadError(str(e)) from e


async def verify_api_key(request: Request) -> None:
    """FastAPI dependency. Raises 401 if auth fails, 503 if auth status
    can't even be determined (fail closed, never fail open).

    On success, records the caller's principal on `request.state.principal`
    so session-scoped routes can enforce ownership (see
    security/ownership.py). Dev mode yields the shared DEV_PRINCIPAL.
    """
    from tokenmizer.security.ownership import DEV_PRINCIPAL, principal_for_key

    try:
        configured = _get_configured_key()
    except _ConfigReadError:
        # We genuinely don't know if a key is required. Refusing the
        # request (503) is the only safe choice — letting it through would
        # mean an unknown number of requests get silently unauthenticated
        # access during whatever window the settings error persists.
        raise HTTPException(
            status_code=503,
            detail="Auth configuration unavailable — request rejected for safety. "
                   "This is a server-side configuration problem, not a client error.",
        )

    if not configured:
        request.state.principal = DEV_PRINCIPAL
        return  # dev mode — auth disabled (explicit empty key, not an error)

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

    # Constant-time comparison against every configured credential.
    # Every candidate is compared (no early break) so that response time
    # does not reveal WHICH key matched, or how many are configured.
    try:
        candidates = _get_configured_keys()
    except Exception:
        candidates = [configured]
    presented = hashlib.sha256(key.encode()).digest()
    valid = False
    for candidate in candidates:
        if hmac.compare_digest(presented, hashlib.sha256(candidate.encode()).digest()):
            valid = True
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid API key")

    request.state.principal = principal_for_key(key)
