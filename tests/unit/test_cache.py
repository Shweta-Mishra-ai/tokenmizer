"""Unit tests — semantic cache with LRU eviction."""
import time

import pytest

from tokenmizer.semantic_cache.cache import SemanticCache


@pytest.fixture
def cache():
    return SemanticCache(threshold=0.95, ttl_seconds=3600, max_size=5)


class TestExactMatch:

    def test_set_and_get(self, cache):
        cache.set("What is Python?", "Python is a programming language.", 10, 20)
        entry = cache.get("What is Python?")
        assert entry is not None
        assert entry.response == "Python is a programming language."

    def test_miss_returns_none(self, cache):
        assert cache.get("something not cached") is None

    def test_hit_increments_count(self, cache):
        cache.set("hello", "world", 1, 1)
        cache.get("hello")
        cache.get("hello")
        entry = cache.get("hello")
        assert entry.hit_count >= 2


class TestLRUEviction:

    def test_eviction_when_full(self, cache):
        """When cache is full, LRU entry is evicted."""
        for i in range(5):
            cache.set(f"prompt {i}", f"response {i}", 1, 1)

        assert len(cache._exact) == 5

        # Access prompt 0 to make it recently used
        cache.get("prompt 0")

        # Add one more — should evict prompt 1 (LRU after 0 was accessed)
        cache.set("new prompt", "new response", 1, 1)

        assert len(cache._exact) == 5  # still at max
        assert cache._eviction_count == 1

    def test_evicted_count_tracked(self, cache):
        for i in range(10):  # way over max_size=5
            cache.set(f"query {i}", f"answer {i}", 1, 1)
        assert cache._eviction_count == 5
        assert len(cache._exact) == 5

    def test_lru_order_correct(self):
        """Oldest unused entry is evicted first."""
        c = SemanticCache(max_size=3)
        c.set("a", "answer a", 1, 1)
        c.set("b", "answer b", 1, 1)
        c.set("c", "answer c", 1, 1)

        # Access a and b — c is now LRU
        c.get("a")
        c.get("b")

        # Add d — c should be evicted
        c.set("d", "answer d", 1, 1)
        assert c.get("c") is None  # evicted
        assert c.get("a") is not None  # still there
        assert c.get("b") is not None  # still there
        assert c.get("d") is not None  # new entry


class TestTTL:

    def test_expired_entry_not_returned(self):
        c = SemanticCache(ttl_seconds=1)
        c.set("expiring prompt", "response", 1, 1)
        # Don't actually sleep in tests — just fake the timestamp
        entry = list(c._exact.values())[0]
        entry.created_at = time.time() - 3600  # 1 hour ago
        result = c.get("expiring prompt")
        assert result is None


class TestStats:

    def test_stats_structure(self, cache):
        cache.set("q1", "a1", 1, 1)
        cache.get("q1")  # hit
        cache.get("unknown")  # miss

        s = cache.stats()
        assert "entries" in s
        assert "hit_rate" in s
        assert "evictions" in s
        assert "utilization_pct" in s
        assert s["entries"] == 1
        assert s["hit_exact"] == 1
        assert s["miss"] == 1
