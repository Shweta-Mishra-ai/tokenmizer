"""
What stops the cache from being the largest thing in the process.

`max_size` is a cap on ENTRIES, and an entry holds a whole LLM response.
Ten thousand entries at 60 KB each is 600 MB resident, reached by a
perfectly ordinary week of code-generation traffic — and the only number
`stats()` reported was `utilization_pct: 100`, which reads as "the cache
is full" rather than as "the cache is a gigabyte". Nothing in the
process would have told an operator which of those two it meant.

Four bounds are pinned here, plus the one invariant the whole scheme
rests on: the byte total must never drift away from the actual contents,
because every eviction decision is made from it. An accounting bug that
undercounts is an unbounded cache wearing a bound.
"""
from __future__ import annotations

from tokenmizer.semantic_cache.cache import SemanticCache


def _cache(**kw) -> SemanticCache:
    kw.setdefault("ttl_seconds", 3600)
    return SemanticCache(**kw)


class TestTheByteBoundIsEnforced:

    def test_a_cache_far_under_its_entry_cap_still_stops_growing(self):
        """The whole point: entries 12, bytes 100%."""
        c = _cache(max_size=10_000, max_bytes=200_000, max_entry_bytes=1_000_000)

        for i in range(200):
            c.set(f"prompt {i}", "x" * 20_000, session_id="s")

        assert c.stats()["bytes"] <= 200_000
        assert len(c._exact) < 20, (
            "entry count is nowhere near its cap; only the byte bound "
            "can have stopped this"
        )

    def test_the_operator_can_see_the_size(self):
        c = _cache(max_bytes=1_000_000)
        c.set("p", "y" * 5000, session_id="s")

        st = c.stats()
        assert st["bytes"] > 5000
        assert st["max_bytes"] == 1_000_000
        assert 0 < st["bytes_pct"] < 100

    def test_eviction_for_bytes_is_counted_separately(self):
        c = _cache(max_size=10_000, max_bytes=60_000)
        for i in range(40):
            c.set(f"p{i}", "z" * 10_000, session_id="s")

        assert c.stats()["evicted_for_bytes"] > 0


class TestOneHugeResponseCannotSpendTheBudget:

    def test_it_is_served_but_not_remembered(self):
        c = _cache(max_entry_bytes=10_000)
        c.set("small", "a" * 100, session_id="s")

        c.set("huge", "b" * 500_000, session_id="s")

        assert c.get("huge", session_id="s") is None
        assert c.get("small", session_id="s") is not None, (
            "a single oversized answer must not evict the useful cache"
        )
        assert c.stats()["rejected_too_large"] == 1, "counted, not silent"


class TestExpiredEntriesAreReclaimed:

    def test_a_dead_entry_nobody_asks_for_is_swept(self):
        """Expiry used to be checked only on lookup, so an entry nobody
        queries again sat in memory until LRU pressure reached it."""
        # -1, not 0: is_expired() is `elapsed > ttl_seconds`, and a tight
        # loop can run entirely inside ONE tick of a coarse wall clock —
        # Windows' time.time() resolution is ~15ms, so `elapsed > 0` was
        # false for entries created in the same tick as the check, and
        # this failed there while passing everywhere else with finer
        # clocks. -1 makes "already expired" true regardless of clock
        # granularity, without changing what code path is exercised.
        c = _cache(ttl_seconds=-1, max_size=10_000)
        for i in range(c._SWEEP_EVERY + 5):
            c.set(f"p{i}", "x" * 200, session_id="s")

        assert c.stats()["expired_swept"] > 0
        assert len(c._exact) <= 10, (
            f"{len(c._exact)} expired entries still resident"
        )


class TestTheAccountingCannotDrift:
    """Every eviction decision is made from `_bytes`. If it drifts low,
    the bound silently stops existing."""

    def _recount(self, c: SemanticCache) -> int:
        return (sum(e.nbytes for e in c._exact.values())
                + len(c._embeddings) * c._EMBEDDING_BYTES)

    def test_overwriting_a_key_does_not_double_count(self):
        c = _cache(max_bytes=10_000_000)
        for _ in range(20):
            c.set("same prompt", "x" * 5000, session_id="s")

        assert len(c._exact) == 1
        assert c._bytes == self._recount(c)

    def test_it_holds_across_sets_evictions_and_invalidations(self):
        c = _cache(max_size=25, max_bytes=400_000)
        for i in range(120):
            c.set(f"p{i}", "x" * (500 + i * 40), session_id="s")
        c.invalidate("p119", session_id="s")

        assert c._bytes == self._recount(c)

    def test_clear_returns_every_byte(self):
        c = _cache()
        for i in range(30):
            c.set(f"p{i}", "x" * 1000, session_id="s")
        c.clear()

        assert c._bytes == 0


class TestTheSemanticScanIsBounded:

    def test_the_miss_path_does_not_walk_the_whole_cache(self):
        """An O(n) Python loop with a dot product per entry, on the path
        a request takes when the cache did NOT help it."""
        c = _cache(max_size=10_000, max_semantic_scan=50)
        assert c.max_semantic_scan == 50

        seen = []

        class _Embedder:
            available = True

            def embed(self, text):
                return [1.0, 0.0]

        c._embedder = _Embedder()
        for i in range(400):
            c.set(f"p{i}", "answer", session_id="s")

        from tokenmizer.semantic_cache.cache import EmbeddingEngine
        original = EmbeddingEngine.cosine

        def counting(a, b):
            seen.append(1)
            return 0.0

        EmbeddingEngine.cosine = staticmethod(counting)
        try:
            c.get("never stored", session_id="s")
        finally:
            EmbeddingEngine.cosine = original

        assert len(seen) <= 50, (
            f"{len(seen)} similarity comparisons for one miss; the bound "
            f"is the only thing standing between this and 10,000"
        )
