"""
Regression tests for the remainder of TM-02 (concurrency / lifecycle safety).

Scope note (corrected from the original audit write-up): a controlled
probe during implementation showed that pure in-process dict mutations on
a SHARED GraphMemory object across concurrent asyncio coroutines do NOT
actually interleave mid-call — CPython's cooperative scheduling only
switches coroutines at `await` points, and GraphMemory's mutation methods
contain no internal awaits. So a blanket "add a lock around every graph
mutation" fix would not have changed anything real. The three actual bugs
this file guards against are narrower and more precise:

  1. `asyncio.create_task(_background_extract())` discarded its return
     value. Per the asyncio docs, a task with no strong reference held
     anywhere can be garbage-collected mid-execution. Fixed by keeping a
     module-level set of in-flight tasks with a done-callback to discard.

  2. `_graph_cache_touch()`'s LRU eviction could evict (and force-persist)
     a GraphMemory instance whose session lock is CURRENTLY HELD — i.e.
     while a request or the background extraction task is actively
     using/about to mutate it. That's a real cross-instance clobber risk:
     if the session is later re-fetched (creating a fresh GraphMemory that
     reloads from whatever's on disk), the in-flight mutation on the old
     evicted instance can persist AFTER the new instance has already
     started writing, silently discarding the newer instance's nodes.
     Fixed the same way `_session_locks`' own eviction already protects
     itself: only evict a session whose lock is unheld.

  3. The genuine lost-update race this codebase has is the accumulator
     pattern (read shared state, await, write back) — see
     test_context_tracking.py for the confirmed, reproduced instance of
     that. That one was a design bug (removed entirely), not a missing
     lock.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from tokenmizer.api import app as app_module
from tokenmizer.graph_memory.graph import GraphMemory


@pytest.fixture(autouse=True)
def clean_caches():
    """Each test gets an empty graph cache / session-lock table so that
    state from one test can't influence another (both are module-level
    singletons in api/app.py)."""
    app_module._graph_cache.clear()
    app_module._session_locks.clear()
    yield
    app_module._graph_cache.clear()
    app_module._session_locks.clear()


class TestBackgroundTaskReferenceRetention:

    async def test_scheduled_background_task_is_tracked_until_done(self, tmp_path, monkeypatch):
        """A scheduled background extraction task must be held by a
        strong reference somewhere in the module until it completes —
        otherwise asyncio may garbage-collect it mid-flight (documented
        asyncio.create_task pitfall)."""
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "use_llm_extraction", True)

        release = asyncio.Event()

        class _SlowFakeProvider:
            async def chat(self, **kwargs):
                await release.wait()
                class R:
                    text = '{"goals": [], "tasks_done": [], "tasks_wip": [], ' \
                           '"tasks_todo": [], "decisions": [], "files": [], ' \
                           '"errors": [], "dependencies": [], "environments": [], ' \
                           '"endpoints": [], "schemas": [], "superseded": []}'
                return R()

        monkeypatch.setattr(app_module, "_get_cheap_provider", lambda: _SlowFakeProvider())

        graph = GraphMemory("bg-task-test", storage_dir=str(tmp_path))
        raw = [{"role": "user", "content": "Use PostgreSQL for the primary datastore please"}]
        messages = [dict(m) for m in raw]

        await app_module._update_graph(
            "bg-task-test", graph, raw, messages, "claude-sonnet-4-6", {}, "irrelevant query",
        )

        # The task must be scheduled AND tracked — not fire-and-forget
        # with zero references, which risks GC before it completes.
        assert len(app_module._background_tasks) == 1, (
            "background extraction task was not retained anywhere — it "
            "is vulnerable to garbage collection before it completes "
            "(see asyncio.create_task's documented pitfall)"
        )

        release.set()  # let the fake provider call return
        # Give the event loop a turn to run the now-unblocked task to completion.
        for _ in range(10):
            if not app_module._background_tasks:
                break
            await asyncio.sleep(0)

        assert len(app_module._background_tasks) == 0, (
            "completed background task was never removed from the "
            "tracking set — it will accumulate forever on a busy server"
        )


class TestGraphCacheEvictionSkipsInFlightSessions:

    async def test_eviction_never_drops_a_session_with_a_held_lock(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_module, "_GRAPH_CACHE_MAX", 3)

        # Fill the cache to capacity with 3 sessions, oldest first.
        for i in range(3):
            g = GraphMemory(f"cache-fill-{i}", storage_dir=str(tmp_path))
            app_module._graph_cache[f"cache-fill-{i}"] = g

        # Hold the lock for the OLDEST (most-LRU) session — simulating an
        # in-flight request that hasn't released it yet.
        oldest_id = next(iter(app_module._graph_cache))
        lock = app_module._get_session_lock(oldest_id)
        await lock.acquire()
        try:
            # Adding a 4th session pushes the cache over the cap and
            # triggers eviction. The oldest entry's lock is held, so
            # eviction must skip it and take the next-oldest instead.
            new_g = await app_module._get_graph_async("cache-fill-new")

            assert oldest_id in app_module._graph_cache, (
                f"session {oldest_id!r} was evicted while its lock was "
                f"held — a request or background task actively using it "
                f"could have its write clobbered by a stale reload"
            )
            assert len(app_module._graph_cache) == 3, (
                "cache exceeded its cap after eviction should have "
                "removed exactly one (unheld) entry"
            )
        finally:
            lock.release()

    async def test_eviction_proceeds_normally_when_nothing_is_held(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_module, "_GRAPH_CACHE_MAX", 3)
        for i in range(3):
            g = GraphMemory(f"cache-normal-{i}", storage_dir=str(tmp_path))
            app_module._graph_cache[f"cache-normal-{i}"] = g
        oldest_id = next(iter(app_module._graph_cache))

        await app_module._get_graph_async("cache-normal-new")

        assert oldest_id not in app_module._graph_cache, (
            "normal LRU eviction (no locks held) should still work exactly "
            "as before — nothing in-flight to protect"
        )
        assert len(app_module._graph_cache) == 3


class TestMultiWorkerRiskWarning:
    """
    Item 4 (see module docstring): multiple OS worker processes each get
    their own _graph_cache and GraphMemory instances for the same
    session_id, and the full-blob SQLite persist means whichever process
    saves last silently overwrites the others' nodes. No in-process lock
    can fix that — issue #27's per-row persistence migration is the real
    fix. Until then, the honest thing to do is warn loudly at startup
    rather than fail silently in production.
    """

    def test_warns_when_a_known_multi_worker_env_var_is_set(self, monkeypatch, caplog):
        import importlib

        from tokenmizer.api import app as app_module

        monkeypatch.setenv("WEB_CONCURRENCY", "4")
        with caplog.at_level(logging.WARNING, logger="tokenmizer.api.app"):
            app_module._warn_if_multi_worker_risk()
        assert any(
            "multi" in r.message.lower() and "process" in r.message.lower()
            for r in caplog.records
        ), "expected a multi-process risk warning when WEB_CONCURRENCY > 1"

    def test_no_warning_for_single_worker(self, monkeypatch, caplog):
        from tokenmizer.api import app as app_module

        for var in ("WEB_CONCURRENCY", "UVICORN_WORKERS", "GUNICORN_WORKERS"):
            monkeypatch.delenv(var, raising=False)
        with caplog.at_level(logging.WARNING, logger="tokenmizer.api.app"):
            app_module._warn_if_multi_worker_risk()
        assert not any("multi" in r.message.lower() and "process" in r.message.lower()
                       for r in caplog.records)

    def test_malformed_env_var_does_not_crash(self, monkeypatch):
        from tokenmizer.api import app as app_module

        monkeypatch.setenv("WEB_CONCURRENCY", "not-a-number")
        app_module._warn_if_multi_worker_risk()  # must not raise
