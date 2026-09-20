"""
What the proxy does when a lot of requests arrive at once.

The background LLM extraction pass was fire-and-forget with nothing
bounding how many could run at the same time. The comment above
`_background_tasks` said the set "never grows unbounded" — true over
time, because each task removes itself, and irrelevant under load: a
burst of N requests produced N concurrent tasks, each holding its slice
of the transcript and each making an upstream call. A memory spike and a
self-inflicted rate limit on the cheap provider, arriving at exactly the
moment the proxy is busiest.

Shedding this particular work is safe in a way shedding most work is
not, and that is the whole reason this is the thing bounded: the
heuristic extraction has ALREADY run synchronously by the time the task
is scheduled, so the graph has the turn's facts either way. What is
dropped is the LLM refinement of one turn.

The counter is deliberately separate from `persist_failures`. "My cheap
provider key expired" and "my proxy is at capacity" lead to opposite
actions — fix the key, or raise the limit — and a single dict of mixed
counts cannot tell an operator which one they have.
"""
from __future__ import annotations

import asyncio

import pytest

from tokenmizer.analytics.engine import AnalyticsEngine


class TestTheExtractionSlotBoundsConcurrency:

    def test_it_declines_rather_than_waits(self):
        """Waiting is the failure mode: a task that waits still holds
        its memory. An asyncio.Semaphore would have queued them."""
        from tokenmizer.api import app as m

        held = [m._extraction_slot() for _ in range(m._EXTRACTION_MAX_CONCURRENT)]
        assert all(cm.__enter__() for cm in held)
        try:
            with m._extraction_slot() as got:
                assert got is False, "over the limit it must decline, not block"
        finally:
            for cm in held:
                cm.__exit__(None, None, None)

    def test_a_slot_is_returned_even_when_the_body_raises(self):
        from tokenmizer.api import app as m

        before = m._extraction_inflight
        with pytest.raises(RuntimeError):
            with m._extraction_slot() as got:
                assert got is True
                raise RuntimeError("upstream blew up")

        assert m._extraction_inflight == before, (
            "a leaked slot permanently lowers the ceiling"
        )

    async def test_a_burst_never_exceeds_the_limit(self):
        """The actual load shape: many coroutines entering at once."""
        from tokenmizer.api import app as m

        peak = 0
        entered = 0

        async def one():
            nonlocal peak, entered
            with m._extraction_slot() as got:
                if not got:
                    return
                entered += 1
                peak = max(peak, m._extraction_inflight)
                await asyncio.sleep(0.01)

        await asyncio.gather(*[one() for _ in range(500)])

        assert peak <= m._EXTRACTION_MAX_CONCURRENT, (
            f"{peak} concurrent extractions against a limit of "
            f"{m._EXTRACTION_MAX_CONCURRENT}"
        )
        assert entered > 0, "the limit must not be a total block"
        assert m._extraction_inflight == 0, "every slot returned"


class TestAShedIsNotAFailure:

    def test_the_two_counters_are_separate(self):
        a = AnalyticsEngine(storage_dir="/tmp/tokenmizer-test-shed")

        a.record_shed("llm_extraction")
        a.record_shed("llm_extraction")
        a.record_silent_failure("llm_extraction")

        assert a.shed == {"llm_extraction": 2}
        assert a.persist_failures == {"llm_extraction": 1}

    def test_health_reports_shedding_without_calling_it_degraded(self):
        """If load shedding turned /health red, an operator would learn
        to ignore the field that means something was actually lost."""
        import inspect

        from tokenmizer.api import app as m

        source = inspect.getsource(m.health)
        assert '"shed": _analytics.shed' in source
        degraded_line = next(ln for ln in source.splitlines()
                             if "degraded = bool(" in ln)
        assert "shed" not in degraded_line
