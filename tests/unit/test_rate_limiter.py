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


@pytest.mark.asyncio
async def test_hard_cap_evicts_oldest_and_keeps_size_bounded():
    """
    Regression test for TM-05's second half: the hard-cap eviction used
    to do `sorted(self._buckets.items(), key=...)` over the WHOLE bucket
    dict while holding the limiter's single global lock — O(n log n) on
    every request once at cap, serializing all traffic behind one sort.
    This only asserts functional correctness (bounded size, LRU-ish
    ordering preserved) since Big-O isn't directly observable from a unit
    test — the O(1)-eviction implementation is verified by code review /
    by this passing with an OrderedDict-based rewrite.
    """
    limiter = RateLimiter(rate=10, per_seconds=60, burst=0, max_clients=5)
    for i in range(20):
        await limiter.check(f"client-{i}")
    assert len(limiter._buckets) <= 5, (
        f"rate limiter exceeded max_clients cap: {len(limiter._buckets)} buckets"
    )
    # Most recently added clients should still be present (oldest evicted first)
    assert "client-19" in limiter._buckets


@pytest.mark.asyncio
async def test_hard_cap_eviction_does_not_evict_recently_touched_client():
    """A client that keeps making requests must not get evicted out from
    under itself just because many OTHER distinct clients show up."""
    limiter = RateLimiter(rate=10, per_seconds=60, burst=0, max_clients=5)
    await limiter.check("loyal-client")
    for i in range(20):
        await limiter.check(f"one-off-{i}")
        await limiter.check("loyal-client")  # touched every round — stays fresh
    assert "loyal-client" in limiter._buckets
