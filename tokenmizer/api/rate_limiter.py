"""
Simple token-bucket rate limiter for the proxy endpoint.
No external deps — pure stdlib.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class _Bucket:
    tokens: float
    last_refill: float = field(default_factory=time.monotonic)


class RateLimiter:
    """
    Per-client token-bucket rate limiter.
    Default: 60 requests/minute per API key (or IP if no key).

    Usage:
        limiter = RateLimiter(rate=60, per_seconds=60, burst=10)
        allowed, retry_after = limiter.check("client-id")
    """

    def __init__(
        self,
        rate: int = 60,
        per_seconds: int = 60,
        burst: int = 10,
        max_clients: int = 50_000,
    ):
        self.rate = rate                  # tokens per window
        self.per_seconds = per_seconds    # window length
        self.burst = burst                # max burst above window rate
        self.max_clients = max_clients    # hard cap — prevents unbounded growth
        self.capacity = rate + burst
        self.refill_rate = rate / per_seconds   # tokens per second
        # OrderedDict, not defaultdict: iteration order tracks LRU
        # (touched-most-recently moves to the end), which is what makes
        # the hard-cap eviction below O(1) rather than a sort.
        self._buckets: "OrderedDict[str, _Bucket]" = OrderedDict()
        self._lock = asyncio.Lock()
        # Cleanup: evict stale buckets every 5 minutes
        self._last_cleanup = time.monotonic()
        self._cleanup_interval = 300

    async def check(self, client_id: str) -> tuple[bool, float]:
        """
        Returns (allowed, retry_after_seconds).
        retry_after is 0.0 if allowed.
        """
        async with self._lock:
            now = time.monotonic()
            if client_id in self._buckets:
                self._buckets.move_to_end(client_id)
                bucket = self._buckets[client_id]
            else:
                bucket = _Bucket(tokens=float(self.capacity))
                self._buckets[client_id] = bucket

            # Refill
            elapsed = now - bucket.last_refill
            bucket.tokens = min(
                self.capacity,
                bucket.tokens + elapsed * self.refill_rate
            )
            bucket.last_refill = now

            # Evict stale buckets periodically
            if now - self._last_cleanup > self._cleanup_interval:
                self._evict_stale(now)

            # Hard cap. Eviction must stay O(1) per entry: this runs
            # while holding the limiter's single global lock, so a sort
            # over the whole bucket dict would let one client sending
            # distinct client_ids stall ALL traffic once at cap.
            # popitem(last=False) drops the LRU entry directly.
            if len(self._buckets) > self.max_clients:
                evict_count = max(1, self.max_clients // 10)
                for _ in range(min(evict_count, len(self._buckets) - 1)):
                    self._buckets.popitem(last=False)
                logger.warning(f"Rate limiter hard cap hit — evicted {evict_count} oldest buckets")

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0
            else:
                # How long until 1 token refills
                retry_after = (1.0 - bucket.tokens) / self.refill_rate
                logger.warning(f"Rate limit hit for client '{client_id}'")
                return False, retry_after

    def _evict_stale(self, now: float, stale_after: float = 600.0) -> None:
        """Remove buckets inactive for >10 minutes to prevent memory leak."""
        stale = [k for k, b in self._buckets.items()
                 if (now - b.last_refill) > stale_after]
        for k in stale:
            del self._buckets[k]
        if stale:
            logger.debug(f"Rate limiter: evicted {len(stale)} stale buckets")
        self._last_cleanup = now


# Singleton
_limiter: RateLimiter | None = None

def get_rate_limiter(rate: int = 60, per_seconds: int = 60, burst: int = 10) -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(rate=rate, per_seconds=per_seconds, burst=burst)
    return _limiter
