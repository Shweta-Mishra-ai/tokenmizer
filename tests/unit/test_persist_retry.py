"""
Regression tests for TM-12: the graph-eviction retry/alerting path was
unreachable because GraphMemory._persist() swallowed its own exceptions
and returned None unconditionally.

Background: api/app.py's _graph_cache_touch() wraps
`evicted_graph._persist()` in a try/except with a retry-once loop and
records a silent-failure metric if both attempts fail — but _persist()
catches Exception INTERNALLY and never re-raises, so that try/except
never actually caught anything. `persisted = True` was set on the very
first call every single time, the retry never ran, and
_analytics.record_silent_failure("graph_eviction") was dead code — an
18-line comment above it described exactly this retry/alerting behavior
as implemented, but it never executed once.

Fix: _persist() now returns bool (True = state is safely on disk, either
because the write succeeded or nothing was dirty; False = a write was
attempted and failed) instead of catching-and-hiding. Callers check the
return value instead of relying on an exception that will never come.
"""
from __future__ import annotations

import pytest

from tokenmizer.graph_memory.graph import GraphMemory, NodeType


@pytest.fixture
def graph(tmp_path):
    return GraphMemory("persist-retry-test", storage_dir=str(tmp_path))


class TestPersistReturnsBool:

    def test_successful_persist_returns_true(self, graph):
        graph.add_node(NodeType.TASK, "Implement the login flow")
        assert graph._persist() is True

    def test_skipped_persist_not_dirty_returns_true(self, graph):
        graph.add_node(NodeType.TASK, "Implement the login flow")
        graph._persist()
        # Nothing changed since — dirty flag is False, so this should
        # skip the write entirely, but the STATE is still correctly on
        # disk (from the prior successful call), so this is success too.
        assert graph._dirty is False
        assert graph._persist() is True

    def test_failed_persist_returns_false(self, graph, monkeypatch):
        def _boom():
            raise sqlite3_error()

        def sqlite3_error():
            import sqlite3
            return sqlite3.OperationalError("simulated disk failure")

        graph.add_node(NodeType.TASK, "Implement the login flow")
        monkeypatch.setattr(graph, "_db_connect", lambda: (_ for _ in ()).throw(
            __import__("sqlite3").OperationalError("simulated disk failure")
        ))
        assert graph._persist() is False
        # And the dirty flag must stay True — a failed write must be
        # retried on the NEXT call, not silently treated as done.
        assert graph._dirty is True

    def test_force_persist_still_returns_bool_correctly(self, graph, monkeypatch):
        graph.add_node(NodeType.TASK, "Implement the login flow")
        graph._persist()
        assert graph._persist(force=True) is True

        monkeypatch.setattr(graph, "_db_connect", lambda: (_ for _ in ()).throw(
            __import__("sqlite3").OperationalError("simulated disk failure")
        ))
        assert graph._persist(force=True) is False


class TestGraphCacheEvictionRetryIsReachable:
    """The api/app.py side of the fix: the retry loop and the
    record_silent_failure metric must actually execute when persist
    keeps failing, and must NOT fire when it eventually succeeds."""

    async def test_eviction_records_silent_failure_when_persist_keeps_failing(
        self, tmp_path, monkeypatch
    ):
        from tokenmizer.api import app as app_module

        monkeypatch.setattr(app_module, "_GRAPH_CACHE_MAX", 1)
        app_module._graph_cache.clear()

        g = GraphMemory("evict-fail-test", storage_dir=str(tmp_path))
        g.add_node(NodeType.TASK, "Something that must not be lost silently")
        app_module._graph_cache["evict-fail-test"] = g

        # Every persist attempt on this instance fails, permanently.
        monkeypatch.setattr(g, "_persist", lambda force=False: False)

        recorded = []
        monkeypatch.setattr(
            app_module._analytics, "record_silent_failure",
            lambda source: recorded.append(source),
        )

        # Adding a new session pushes the cache over cap and evicts the
        # (unlocked) 'evict-fail-test' entry, attempting to persist it.
        await app_module._get_graph_async("evict-fail-test-2")

        assert "graph_eviction" in recorded, (
            "persist kept failing on eviction but record_silent_failure "
            "was never called — the retry/alerting path is still dead code"
        )

    async def test_eviction_does_not_record_failure_when_persist_succeeds(
        self, tmp_path, monkeypatch
    ):
        from tokenmizer.api import app as app_module

        monkeypatch.setattr(app_module, "_GRAPH_CACHE_MAX", 1)
        app_module._graph_cache.clear()

        g = GraphMemory("evict-ok-test", storage_dir=str(tmp_path))
        g.add_node(NodeType.TASK, "Something that persists just fine")
        app_module._graph_cache["evict-ok-test"] = g

        recorded = []
        monkeypatch.setattr(
            app_module._analytics, "record_silent_failure",
            lambda source: recorded.append(source),
        )

        await app_module._get_graph_async("evict-ok-test-2")

        assert "graph_eviction" not in recorded


class TestInvalidateDecisionSurfacesPersistFailure:
    """The /api/decision/invalidate endpoint mutates node.status directly
    then force-persists. It must not claim success in the HTTP response
    if that persist actually failed — that would be the exact silent-
    data-loss pattern (claim success, lose the write) this whole audit
    targets, just at the API layer instead of the storage layer."""

    def test_returns_500_when_persist_fails(self, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock

        from fastapi.testclient import TestClient

        from tokenmizer.api import app as app_module
        from tokenmizer.api.app import app
        from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType

        g = GraphMemory("invalidate-persist-fail", storage_dir=str(tmp_path))
        g.add_node(NodeType.DECISION, "Use PostgreSQL for storage", NodeStatus.COMPLETED)
        app_module._graph_cache["invalidate-persist-fail"] = g
        monkeypatch.setattr(g, "_persist", lambda force=False: False)
        monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)

        with TestClient(app) as c:
            r = c.post(
                "/api/decision/invalidate",
                params={"session_id": "invalidate-persist-fail",
                        "decision_label": "postgresql"},
            )
        assert r.status_code == 500
        assert "did not persist" in r.json()["detail"] or "FAILED" in r.json()["detail"]
        # The in-memory mutation still happened — the bug isn't that the
        # status flip didn't happen, it's that we must not CLAIM success.
        assert g._nodes[next(iter(g._nodes))].status == NodeStatus.INVALIDATED
