"""
Cross-process advisory file lock.

WHY THIS EXISTS
---------------
Session state is guarded in-process by an asyncio.Lock, which does
nothing across process boundaries. Two uvicorn workers (or a worker and
the CLI, or a worker and a cron job) holding the same session each keep
their own GraphMemory instance, and each writes what IT believes the
graph contains.

Per-row storage already stopped that from being catastrophic — disjoint
node additions merge instead of one writer's blob replacing the other's.
What it does not fix is staleness:

    worker A loads a session (100 nodes)
    worker B loads the same session (100 nodes)
    worker A prunes it to 50 and persists      -> 50 rows on disk
    worker B adds one node and persists        -> its 100 nodes come back

B is not misbehaving; it is faithfully writing the state it holds. The
only fix is to make read-modify-write atomic across processes, which
needs a lock the OS enforces, not one the interpreter does.

SCOPE AND LIMITS
----------------
- Advisory, and POSIX-record-lock based (`fcntl.flock`) on Unix,
  `msvcrt.locking` on Windows. Both are per-file and released
  automatically if the holding process dies, which matters: a crashed
  worker must not wedge a session forever.
- The lock file lives next to the database, one per session, so two
  sessions never contend with each other.
- flock semantics on NFS are unreliable. A storage_dir on NFS is not
  supported for multi-process use; this is documented rather than
  silently depended upon.
- This orders writers. It does not make analytics, the rate limiter or
  the semantic cache cross-process — those remain per-process.
"""
from __future__ import annotations

import errno
import logging
import time
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

try:  # POSIX
    import fcntl
    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover — Windows
    fcntl = None  # type: ignore[assignment]
    _HAVE_FCNTL = False

try:  # Windows
    import msvcrt
    _HAVE_MSVCRT = True
except ImportError:
    msvcrt = None  # type: ignore[assignment]
    _HAVE_MSVCRT = False


class LockUnavailable(Exception):
    """The lock could not be acquired within the timeout."""


def _safe_name(session_id: str) -> str:
    """Filesystem-safe lock filename for a session.

    session_id is client-supplied, so it can contain path separators,
    NULs, or be long enough to blow the filename limit. Anything outside
    a conservative allowlist is replaced, and a hash suffix keeps two
    different ids from colliding onto one lock file after sanitising
    (which would make two unrelated sessions block each other).
    """
    import hashlib
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "_" for c in session_id)[:64]
    digest = hashlib.sha256(session_id.encode()).hexdigest()[:12]
    return f"{cleaned}.{digest}.lock"


@contextmanager
def session_lock(storage_dir, session_id: str, timeout: float = 10.0):
    """Hold an exclusive cross-process lock for `session_id`.

    Raises LockUnavailable if it cannot be acquired within `timeout`.
    Callers treat that as "do not write" rather than "write anyway" —
    proceeding without the lock is precisely the corruption this exists
    to prevent.

    If neither locking primitive is available (an unusual platform), the
    lock degrades to a no-op and logs once, because failing every write
    on such a platform would be worse than the single-process behaviour
    that shipped before this existed.
    """
    lock_dir = Path(storage_dir) / ".locks"
    try:
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / _safe_name(session_id)
        fh = open(lock_path, "a+b")
    except OSError as e:
        logger.warning(
            "Could not open lock file for session %s (%s) — proceeding "
            "without cross-process locking for this write.", session_id, e
        )
        yield False
        return

    acquired = False
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                if _HAVE_FCNTL:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                elif _HAVE_MSVCRT:  # pragma: no cover — Windows
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:  # pragma: no cover — neither primitive
                    logger.warning(
                        "No file-locking primitive on this platform; "
                        "multi-process writes are NOT serialised."
                    )
                    yield False
                    return
                acquired = True
                break
            except OSError as e:
                if e.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    raise LockUnavailable(
                        f"Could not acquire the write lock for session "
                        f"{session_id!r} within {timeout}s — another process "
                        f"is holding it."
                    ) from e
                time.sleep(0.05)

        yield True
    finally:
        if acquired:
            try:
                if _HAVE_FCNTL:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                elif _HAVE_MSVCRT:  # pragma: no cover — Windows
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError as e:
                logger.error("Failed to release lock for %s: %s", session_id, e)
        try:
            fh.close()
        except OSError:
            pass


def lock_files_in(storage_dir) -> list:
    """Lock files present in a storage dir. Diagnostic helper only —
    the presence of a file says nothing about whether it is held, since
    the OS releases the lock when a process exits without deleting it."""
    d = Path(storage_dir) / ".locks"
    return sorted(d.glob("*.lock")) if d.exists() else []


__all__ = ["session_lock", "LockUnavailable", "lock_files_in"]
