"""
Session ownership — who is allowed to read or mutate a given session.

THE PROBLEM THIS CLOSES
-----------------------
Every session-scoped route takes `session_id` straight from the URL and
returns (or mutates) that session's data:

    GET  /api/graph/{session_id}          -> the whole knowledge graph
    GET  /api/graph/{session_id}/viz      -> every node and edge
    GET  /api/resume/{session_id}         -> the resume context block
    GET  /api/graph/{session_id}/transitions
    POST /api/checkpoint?session_id=...
    POST /api/decision/invalidate?session_id=...

Authentication proved only that the caller held *the* deployment key;
nothing tied a session to a caller. Any authenticated client could read
any other client's session — and `session_id` is not a secret to guess
at, because clients choose it themselves in the chat request body
(`ChatRequest.session_id`), so "pick a plausible name" was enough. The
invalidate route made it a write primitive too.

THE MODEL
---------
A session is claimed by the first principal that uses it, and only that
principal may touch it afterwards. A principal is derived from the
presented credential, so:

  * Dev mode (no key configured) — every caller is the same principal
    (`__dev__`). Local single-user use is unchanged.
  * One shared key — every caller is the same principal. Behaviour is
    unchanged; this is a single-tenant deployment and is now honestly
    labelled as one.
  * Multiple keys (`api_keys` in settings) — each key is its own
    principal, and sessions are genuinely isolated between them.

Ownership rows are keyed by session_id, stored next to the graph and
checkpoint databases in storage_dir. Claiming is atomic (INSERT OR
IGNORE followed by a read-back), so two concurrent first-requests for
the same session can't both believe they own it.

FAIL-CLOSED
-----------
If the ownership store cannot be read or written, access is DENIED
rather than allowed. An ownership check that fails open is not an
access control — it's a comment. This matches the same reasoning already
applied to `verify_api_key` when settings can't be read.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Principal used when no API key is configured at all (dev mode). A fixed
# sentinel rather than a random value so ownership survives restarts.
DEV_PRINCIPAL = "__dev__"


class SessionAccessDenied(Exception):
    """Raised when a principal touches a session owned by someone else."""


class OwnershipUnavailable(Exception):
    """Raised when ownership state can't be determined — deny, never allow."""


def principal_for_key(api_key: str) -> str:
    """Stable, non-reversible principal id for a credential.

    Hashed so the ownership table never stores raw API keys: leaking that
    file must not leak credentials.
    """
    if not api_key:
        return DEV_PRINCIPAL
    return "k_" + hashlib.sha256(api_key.encode()).hexdigest()[:16]


class OwnershipStore:
    """SQLite-backed session -> owner map."""

    def __init__(self, storage_dir: str = "./checkpoints"):
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._dir / "sessions.db"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=5.0, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            conn.close()
            raise
        return conn

    def _init_db(self) -> None:
        try:
            conn = self._connect()
            try:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS session_owners (
                           session_id TEXT PRIMARY KEY,
                           owner      TEXT NOT NULL,
                           created_at REAL NOT NULL
                       )"""
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            # Deliberately NOT quarantining/deleting anything here: unlike
            # the graph and checkpoint DBs, losing this file would silently
            # re-open every session to whoever asks first. Surface it and
            # let the fail-closed path in check_access() deny requests.
            logger.error(f"Session ownership store unavailable ({self._db_path}): {e}")

    def claim(self, session_id: str, principal: str) -> str:
        """Claim `session_id` for `principal` if unowned. Returns the
        actual owner (which may be someone else). Atomic."""
        try:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO session_owners (session_id, owner, created_at) "
                    "VALUES (?, ?, ?)",
                    (session_id, principal, time.time()),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT owner FROM session_owners WHERE session_id=?", (session_id,)
                ).fetchone()
            finally:
                conn.close()
        except Exception as e:
            raise OwnershipUnavailable(str(e)) from e
        if row is None:
            raise OwnershipUnavailable("owner row vanished immediately after claim")
        return row[0]

    def owner_of(self, session_id: str) -> Optional[str]:
        """Current owner, or None if the session has never been claimed."""
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT owner FROM session_owners WHERE session_id=?", (session_id,)
                ).fetchone()
            finally:
                conn.close()
        except Exception as e:
            raise OwnershipUnavailable(str(e)) from e
        return row[0] if row else None

    def check_access(self, session_id: str, principal: str, *, claim: bool = True) -> None:
        """Raise SessionAccessDenied if `principal` may not touch this
        session. With claim=True an unowned session is claimed.

        claim=False is for read-only inspection routes: reading a session
        that does not exist yet should 404 through the normal path rather
        than silently creating an ownership record as a side effect of a
        GET.
        """
        if claim:
            owner = self.claim(session_id, principal)
        else:
            owner = self.owner_of(session_id)
            if owner is None:
                return  # never claimed — nothing to protect yet
        if owner != principal:
            raise SessionAccessDenied(session_id)
