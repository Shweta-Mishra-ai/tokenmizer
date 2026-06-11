"""
Simple token-bucket rate limiter for the proxy endpoint.
No external deps — pure stdlib.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
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

    def __init__(self, rate: int = 60, per_seconds: int = 60, burst: int = 10):
        self.rate = rate                  # tokens per window
        self.per_seconds = per_seconds    # window length
        self.burst = burst                # max burst above window rate
        self.capacity = rate + burst
        self.refill_rate = rate / per_seconds   # tokens per second
        self._buckets: dict[str, _Bucket] = defaultdict(
            lambda: _Bucket(tokens=float(self.capacity))
        )
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
            bucket = self._buckets[client_id]

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
