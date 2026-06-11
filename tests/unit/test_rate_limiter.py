"""Tests for the rate limiter."""
import pytest

from tokenmizer.api.rate_limiter import RateLimiter


@pytest.mark.asyncio
async def test_allows_under_limit():
    limiter = RateLimiter(rate=10, per_seconds=60, burst=2)
    allowed, retry = await limiter.check("client-1")
    assert allowed is True
    assert retry == 0.0


@pytest.mark.asyncio
async def test_blocks_over_limit():
    limiter = RateLimiter(rate=3, per_seconds=60, burst=0)
    for _ in range(3):
        await limiter.check("client-2")
    allowed, retry = await limiter.check("client-2")
    assert allowed is False
    assert retry > 0


@pytest.mark.asyncio
async def test_different_clients_isolated():
    limiter = RateLimiter(rate=2, per_seconds=60, burst=0)
    for _ in range(2):
        await limiter.check("client-a")
    # client-a is exhausted
    allowed_a, _ = await limiter.check("client-a")
    # client-b should still be allowed
    allowed_b, _ = await limiter.check("client-b")
    assert allowed_a is False
    assert allowed_b is True


@pytest.mark.asyncio
async def test_evicts_stale_buckets():
    limiter = RateLimiter(rate=10, per_seconds=60, burst=0)
    for i in range(5):
        await limiter.check(f"client-{i}")
    # Force cleanup
    import time
    for bucket in limiter._buckets.values():
        bucket.last_refill -= 700  # make them stale
    limiter._evict_stale(time.monotonic(), stale_after=600)
    assert len(limiter._buckets) == 0
