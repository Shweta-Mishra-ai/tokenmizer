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
from tokenmizer.graph_memory.graph import GraphMemory
from tokenmizer.providers.providers import OllamaProvider


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


class TestTheTaskIsNotEvenBuiltWhenFull:
    """The first version of this cap checked capacity INSIDE the task.

    The ceiling on concurrency was real, and the bound on memory — the
    reason the cap exists — was not: a burst of N requests still created
    N coroutine objects, each closing over that turn's whole transcript,
    and each shed itself only once the loop got round to running it. The
    spike the cap was written to prevent happened anyway, one layer down.
    """

    def test_capacity_counts_queued_tasks_as_well_as_running_ones(self):
        from tokenmizer.api import app as m

        assert m._extraction_has_capacity() is True

        m._extraction_pending = m._EXTRACTION_MAX_CONCURRENT
        try:
            assert m._extraction_has_capacity() is False, (
                "a task that has been created but not yet started still "
                "holds its transcript; it is not free"
            )
        finally:
            m._extraction_pending = 0

    def test_running_and_queued_are_counted_together(self):
        from tokenmizer.api import app as m

        half = m._EXTRACTION_MAX_CONCURRENT // 2
        m._extraction_pending = half
        held = [m._extraction_slot() for _ in range(m._EXTRACTION_MAX_CONCURRENT - half)]
        for cm in held:
            cm.__enter__()
        try:
            assert m._extraction_has_capacity() is False
        finally:
            for cm in held:
                cm.__exit__(None, None, None)
            m._extraction_pending = 0

    def test_the_scheduler_checks_before_it_builds_the_coroutine(self):
        """Pinned against the source, because the distinction is one line
        and the difference is the whole point of the cap."""
        import inspect

        from tokenmizer.api import app as m

        source = inspect.getsource(m._update_graph)
        gate = source.index("_extraction_has_capacity()")
        build = source.index("_track_background_task(_background_extract())")
        assert gate < build, (
            "capacity must be tested before the coroutine is created, not "
            "inside it"
        )


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


class TestTheCounterComesBackDown:
    """A counter that only goes up is worse than no counter: extraction
    would be shed for the life of the process, and every /api/stats shed
    count would look like sustained load that is not there."""

    async def test_a_completed_extraction_releases_its_reservation(
            self, tmp_path, monkeypatch):
        from tokenmizer.api import app as m

        monkeypatch.setattr(m.settings, "provider", "ollama")
        monkeypatch.setattr(m.settings, "default_model", "llama3.1:8b")
        monkeypatch.setattr(m.settings.graph_checkpoint,
                            "use_llm_extraction", True)
        m._extraction_provider = None
        m._graph_cache.clear()
        m._session_locks.clear()
        m._extraction_pending = 0

        class _R:
            text = ('{"goals": [], "tasks_done": [], "tasks_wip": [], '
                    '"tasks_todo": [], "decisions": [], "files": [], '
                    '"errors": [], "dependencies": [], "environments": [], '
                    '"endpoints": [], "schemas": [], "superseded": []}')

        async def chat(self, **kwargs):
            return _R()

        monkeypatch.setattr(OllamaProvider, "chat", chat)

        for i in range(3):
            graph = GraphMemory(f"pending-{i}", storage_dir=str(tmp_path))
            raw = [{"role": "user", "content": f"question number {i} about orders"}]
            await m._update_graph(f"pending-{i}", graph, raw,
                                  [dict(x) for x in raw], "llama3.1:8b", {},
                                  "orders")

        for _ in range(200):
            if not m._background_tasks:
                break
            await asyncio.sleep(0)

        assert m._extraction_pending == 0, (
            f"{m._extraction_pending} reservations never released — "
            f"extraction would be shed for the life of the process"
        )
        assert m._extraction_inflight == 0
