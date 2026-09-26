"""
The decisive-token guard runs on the request path. It must not be a way
to make the proxy slow.

Two defects, both found by an audit after the guard shipped, both on the
path a request takes when the cache does NOT help it:

1. `_decisive_profile(prompt)` was called inside the candidate loop, so
   the QUERY was re-profiled on every candidate — up to
   `max_semantic_scan` (2000) times per lookup, every one of them
   identical work. Each CANDIDATE was also re-profiled on every lookup
   that reached it, although a stored prompt never changes.

2. `_LITERAL` was quadratic. Unbounded repeats in front of a required
   literal (`[\\w-]+\\.` then an extension) make the engine rescan the
   run from every start position, and `\\w+(?:_\\w+)+` nests two repeats
   over overlapping sets — `\\w` already contains "_" — which is the
   textbook catastrophic shape.

Together: one cache miss with a 1 KB prompt and 2000 entries took
0.421 s of blocking CPU inside the event loop. Measured, not estimated.
It is 0.001 s now.

These tests pin the ALGORITHM, by counting calls, rather than the clock,
which would flake on a loaded runner. The one timing assertion carries a
margin wide enough that only a return to quadratic can trip it.
"""
from __future__ import annotations

import time

import pytest

from tokenmizer.semantic_cache import cache as cache_mod
from tokenmizer.semantic_cache.cache import EmbeddingEngine, SemanticCache


class _AlwaysScan:
    """Embedder that never scores a hit, so every lookup walks the whole
    candidate list — the expensive path, which is the one under test."""

    available = True

    def embed(self, text):
        return [1.0]


@pytest.fixture
def cache(monkeypatch):
    monkeypatch.setattr(EmbeddingEngine, "cosine", staticmethod(lambda a, b: 0.1))
    c = SemanticCache(threshold=0.9, ttl_seconds=3600, max_size=5000)
    monkeypatch.setattr(c, "_embedder", _AlwaysScan())
    return c


@pytest.fixture
def counted(monkeypatch):
    """Count real profile computations, wherever they are triggered."""
    calls = []
    real = cache_mod._decisive_profile

    def spy(prompt):
        calls.append(prompt)
        return real(prompt)

    monkeypatch.setattr(cache_mod, "_decisive_profile", spy)
    return calls


def _fill(c, n):
    for i in range(n):
        c.set(f"please refactor the handler and explain why, case {i}",
              "answer", session_id="s")


class TestTheQueryIsProfiledOncePerLookup:
    def test_not_once_per_candidate(self, cache, counted):
        _fill(cache, 50)
        counted.clear()

        cache.get("please refactor something else entirely", session_id="s")

        query_profiles = [p for p in counted
                          if p == "please refactor something else entirely"]
        assert len(query_profiles) == 1, (
            f"the query was profiled {len(query_profiles)} times for one "
            f"lookup; it must be profiled once and compared against each "
            f"candidate"
        )


class TestACandidateIsProfiledOnceEver:
    def test_a_second_lookup_reprofiles_nothing(self, cache, counted):
        _fill(cache, 50)
        cache.get("warm the memoised profiles", session_id="s")
        counted.clear()

        cache.get("a different question altogether", session_id="s")

        assert len(counted) == 1, (
            f"{len(counted)} profiles computed on a warm lookup; only the "
            f"query should need one — stored prompts never change"
        )

    def test_the_memoised_profile_matches_a_fresh_one(self, cache):
        """Memoising must not change the answer."""
        cache.set("roll back migration 003", "answer", session_id="s")
        entry = next(iter(cache._exact.values()))

        assert entry._profile is None
        memoised = cache_mod._profile_of(entry)
        assert memoised == cache_mod._decisive_profile("roll back migration 003")
        assert cache_mod._profile_of(entry) is memoised

    def test_memoising_does_not_weaken_the_guard(self, cache):
        """The whole point of the guard still holds on a warm cache."""
        cache.set("how do I enable the cache", "STORED", session_id="s")
        cache.get("warm it", session_id="s")

        assert cache.get("how do I disable the cache", session_id="s") is None


class TestTheProfilePatternIsNotQuadratic:
    """A quadratic pattern is a denial of service on a proxy: the text is
    whatever a caller sent. Generous bounds — only a return to quadratic
    trips these."""

    @pytest.mark.parametrize("pattern_name,module", [
        ("_LITERAL", "tokenmizer.semantic_cache.cache"),
        ("_INFORMATION", "tokenmizer.compression.output_trimmer"),
    ])
    def test_doubling_the_input_does_not_quadruple_the_time(
            self, pattern_name, module):
        """Sized so the measurement survives a shared CI runner.

        A ratio of two timings is only as stable as the smaller one. At
        4 KB the linear pattern takes ~2 ms, where a scheduler hiccup is
        a large fraction of the reading and the ratio can swing past any
        threshold for no reason. 16 KB and 32 KB take ~8 ms and ~16 ms,
        so the same absolute jitter is proportionally small — and the
        quadratic version took 0.79 s at 16 KB, so the signal this is
        looking for is three orders of magnitude clear of the noise.

        min-of-5 rather than a mean: the fastest run is the one least
        interrupted, and interruptions only ever add time.
        """
        import importlib

        pattern = getattr(importlib.import_module(module), pattern_name)

        def took(n):
            text = "a-" * n
            pattern.search(text)                    # warm the cache
            return min(_timed(pattern, text) for _ in range(5))

        small = took(8000)
        large = took(16000)

        # Linear is 2.0x. Quadratic is 4.0x, and was measured at 4.0x.
        # 3.0 sits between them with room on both sides.
        assert large < small * 3.0, (
            f"doubling the input multiplied the time by {large / small:.1f} "
            f"({small * 1000:.1f} ms -> {large * 1000:.1f} ms) — "
            f"{pattern_name} looks super-linear again"
        )

    def test_a_large_prompt_profiles_promptly(self):
        """32 KB took 1.6 s before the bound; the ceiling here is wide
        enough for a slow shared runner and still far below that."""
        started = time.perf_counter()
        cache_mod._decisive_profile("a-" * 16000)
        assert time.perf_counter() - started < 1.0


def _timed(pattern, text) -> float:
    started = time.perf_counter()
    pattern.search(text)
    return time.perf_counter() - started
