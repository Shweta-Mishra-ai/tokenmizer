"""
A token bucket every worker shares.

`RateLimiter` holds its buckets in a process dict, which is correct for
one process and wrong for the deployment the Dockerfile ships: with
`--workers 4`, a configured 60 requests per minute is enforced four times
over, once per worker, and the operator gets 240. A limit that is not the
limit is worse than no limit, because it is written down.

This keeps the buckets in SQLite, in the same storage directory as
everything else durable, so the workers share one count. The whole
read-modify-write runs inside a single `BEGIN IMMEDIATE` transaction:
without that, two workers both read the last token, both decide they may
spend it, and both allow the request — the classic lost update, and here
it is the rate limit itself that is lost.

**When to use which.** `state_backend: memory` (the default) is right for
one process, and costs nothing per request. `state_backend: sqlite` is
right for a multi-worker deployment, and costs one small transaction per
request — measured at well under a millisecond on local disk, against an
LLM call measured in seconds.

**What this is not.** It is not a distributed rate limiter for several
machines: SQLite is shared through the filesystem, so it is shared by the
workers on one host. Several hosts behind a load balancer need the limit
at the load balancer, and this module says so rather than implying
otherwise.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# How long a failed init is believed before trying again. Long enough that
# a genuinely broken file is not reopened on every request, short enough
# that a startup race does not cost a worker its rate limiting for hours.
_RETRY_INIT_AFTER = 30.0


def now_monotonic() -> float:
    return time.monotonic()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rate_buckets (
    client_id   TEXT PRIMARY KEY,
    tokens      REAL NOT NULL,
    last_refill REAL NOT NULL
)
"""


class SQLiteRateLimiter:
    """The same token bucket as `RateLimiter`, shared across processes.

    The interface is identical — `await check(client_id)` returning
    `(allowed, retry_after)` — so `app.py` does not know which one it has.
    """

    def __init__(self, rate: int = 60, per_seconds: int = 60, burst: int = 10,
                 storage_dir: str = "./checkpoints",
                 max_clients: int = 50_000):
        self.rate = rate
        self.per_seconds = per_seconds
        self.burst = burst
        self.capacity = rate + burst
        self.refill_rate = rate / per_seconds
        self.max_clients = max_clients
        self._db_path = Path(storage_dir) / "rate_limits.db"
        self._last_cleanup = 0.0
        self._cleanup_interval = 300.0
        self._unavailable_since = 0.0
        # A failure to open the shared store must not take the proxy with
        # it, and must not silently become "no rate limiting" either: the
        # caller falls back to the in-process limiter, which is the old
        # behaviour rather than none.
        self.available = self._init()
        if not self.available:
            self._unavailable_since = now_monotonic()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=5.0,
                               isolation_level=None, check_same_thread=False)
        try:
            # WAL takes a brief exclusive lock to switch mode, so with four
            # workers starting at once one of them loses it. The mode is a
            # property of the FILE and persists once anybody has set it, so
            # losing the race is not a failure — treating it as one is how
            # a worker used to disable its own rate limiting for good.
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError as e:
            logger.debug("journal_mode=WAL not set on %s (%s) — another "
                         "process is setting it", self._db_path, e)
        try:
            conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.OperationalError:
            pass
        return conn

    def _init(self) -> bool:
        """Create the table, retrying while the file is merely busy.

        `CREATE TABLE IF NOT EXISTS` takes a write lock, and every worker
        runs it at startup — so on a cold start with `--workers 4`, three
        of them can meet a locked database. Treating that first failure as
        final meant a worker marked itself unavailable FOR ITS WHOLE LIFE
        and silently enforced no limit at all: found by the four-process
        test failing under a loaded machine, allowing 20 of a budget of
        10. "Locked" is contention; only a file that stays unusable after
        backing off is broken.
        """
        delay = 0.05
        last: Exception | None = None
        for _ in range(6):
            try:
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
                with self._connect() as conn:
                    conn.execute(_SCHEMA)
                return True
            except sqlite3.OperationalError as e:
                last = e
                time.sleep(delay)
                delay = min(delay * 2, 0.8)
            except Exception as e:
                last = e
                break
        logger.error(
            "Shared rate limiter could not open %s (%s). Falling back to "
            "the per-process limiter: with more than one worker the "
            "configured limit will be enforced once per worker.",
            self._db_path, last,
        )
        return False

    async def check(self, client_id: str) -> tuple[bool, float]:
        """`(allowed, retry_after_seconds)`; retry_after is 0.0 if allowed."""
        if not self.available:
            # One failed init is not a life sentence. A database that was
            # locked at startup is usually fine seconds later, and the
            # alternative is a worker that never rate-limits again.
            if now_monotonic() - self._unavailable_since < _RETRY_INIT_AFTER:
                return True, 0.0
            self.available = self._init()
            if not self.available:
                self._unavailable_since = now_monotonic()
                return True, 0.0

        # Wall clock, not monotonic: monotonic is per-process and these
        # timestamps are compared across processes. The cost is that a
        # clock jump backwards stalls refills until the clock catches up,
        # which is the lesser of the two errors — the other one hands out
        # free tokens.
        now = time.time()
        try:
            conn = self._connect()
        except Exception as e:
            logger.error("Shared rate limiter unavailable mid-run: %s", e)
            return True, 0.0

        try:
            # IMMEDIATE takes the write lock up front, so the read below
            # cannot be overtaken between read and write.
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT tokens, last_refill FROM rate_buckets WHERE client_id = ?",
                (client_id,),
            ).fetchone()

            if row is None:
                tokens, last_refill = float(self.capacity), now
            else:
                tokens, last_refill = float(row[0]), float(row[1])
                tokens = min(self.capacity,
                             tokens + max(0.0, now - last_refill) * self.refill_rate)
                last_refill = now

            if tokens >= 1.0:
                tokens -= 1.0
                allowed, retry_after = True, 0.0
            else:
                allowed = False
                retry_after = (1.0 - tokens) / self.refill_rate

            conn.execute(
                "INSERT INTO rate_buckets (client_id, tokens, last_refill) "
                "VALUES (?, ?, ?) ON CONFLICT(client_id) DO UPDATE SET "
                "tokens = excluded.tokens, last_refill = excluded.last_refill",
                (client_id, tokens, last_refill),
            )
            conn.execute("COMMIT")
        except Exception as e:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            # Fail OPEN, and say so. A limiter that fails closed turns a
            # locked database file into a total outage; one that fails
            # open lets traffic through while an operator can still see
            # the error. This is the same trade the graph makes when
            # persistence breaks: serve the request, report the fault.
            logger.error(
                "Shared rate limiter check failed for %r — allowing the "
                "request unlimited: %s", client_id, e,
            )
            return True, 0.0
        finally:
            conn.close()

        if not allowed:
            logger.warning("Rate limit hit for client '%s'", client_id)
        self._maybe_evict(now)
        return allowed, retry_after

    def _maybe_evict(self, now: float, stale_after: float = 600.0) -> None:
        """Drop buckets nobody has touched, and cap the table.

        A bucket at full capacity is indistinguishable from no bucket at
        all, so forgetting an idle client costs nothing and an unbounded
        table costs an operator their disk.
        """
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM rate_buckets WHERE last_refill < ?",
                             (now - stale_after,))
                over = conn.execute(
                    "SELECT COUNT(*) FROM rate_buckets").fetchone()[0]
                if over > self.max_clients:
                    conn.execute(
                        "DELETE FROM rate_buckets WHERE client_id IN ("
                        "  SELECT client_id FROM rate_buckets "
                        "  ORDER BY last_refill ASC LIMIT ?)",
                        (over - self.max_clients,),
                    )
                    logger.warning(
                        "Shared rate limiter hard cap hit — dropped %d oldest "
                        "buckets", over - self.max_clients,
                    )
        except Exception as e:
            logger.debug("Shared rate limiter cleanup skipped: %s", e)
