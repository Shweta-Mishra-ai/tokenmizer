"""
Regression tests: the checkpoint/invalidate/resume admin endpoints in
api/routes_graph.py used to put raw exception text (`str(e)`) directly
into the HTTP 500 response body — the same silent-internal-detail-leak
class of bug chat_completions()'s provider-failure handler was already
fixed against (TM-33), just not mirrored here. Fixed via a shared
`_internal_error()` helper: full exception text goes to the server log
with a correlation id, the client gets a generic message plus that id.
"""
from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from tokenmizer.api import app as app_module
from tokenmizer.api.app import app
from tokenmizer.api.routes_graph import _internal_error
from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType


class TestInternalErrorHelper:

    def test_detail_does_not_contain_raw_exception_text(self, caplog):
        secret_looking_error = RuntimeError(
            "disk I/O error at /var/secret/prod-db/graph_memory.db"
        )
        with caplog.at_level(logging.ERROR):
            exc = _internal_error("Manual checkpoint failed for session xyz", secret_looking_error)
        assert "/var/secret/prod-db" not in str(exc.detail)
        assert exc.status_code == 500

    def test_full_detail_is_still_logged_server_side(self, caplog):
        secret_looking_error = RuntimeError("disk I/O error at /var/secret/prod-db/graph_memory.db")
        with caplog.at_level(logging.ERROR):
            _internal_error("Manual checkpoint failed for session xyz", secret_looking_error)
        assert any("/var/secret/prod-db" in r.message for r in caplog.records), (
            "the real error must still be logged server-side even though "
            "it's kept out of the client-facing response"
        )

    def test_response_includes_a_correlation_id_reference(self):
        exc = _internal_error("Resume failed for session xyz", ValueError("boom"))
        assert "ref:" in exc.detail


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)


class TestEndpointsDoNotLeakExceptionText:

    def test_checkpoint_endpoint_does_not_leak_exception_text(self, tmp_path, monkeypatch):
        g = GraphMemory("leak-test-checkpoint", storage_dir=str(tmp_path))
        app_module._graph_cache["leak-test-checkpoint"] = g

        def _boom(*a, **k):
            raise RuntimeError("connection string password=hunter2 leaked here")

        monkeypatch.setattr(app_module._checkpoint_mgr, "create", _boom)

        with TestClient(app) as c:
            r = c.post("/api/checkpoint", params={"session_id": "leak-test-checkpoint"})
        assert r.status_code == 500
        assert "hunter2" not in r.text

    def test_resume_endpoint_does_not_leak_exception_text(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("connection string password=hunter2 leaked here")

        monkeypatch.setattr(app_module._checkpoint_mgr, "get_latest", _boom)

        with TestClient(app) as c:
            r = c.get("/api/resume/leak-test-resume")
        assert r.status_code == 500
        assert "hunter2" not in r.text

    def test_invalidate_decision_endpoint_does_not_leak_exception_text(self, tmp_path, monkeypatch):
        g = GraphMemory("leak-test-invalidate", storage_dir=str(tmp_path))
        g.add_node(NodeType.DECISION, "Use PostgreSQL for storage", NodeStatus.COMPLETED)
        app_module._graph_cache["leak-test-invalidate"] = g

        def _boom(force=False):
            raise RuntimeError("connection string password=hunter2 leaked here")

        monkeypatch.setattr(g, "_persist", _boom)

        with TestClient(app) as c:
            r = c.post(
                "/api/decision/invalidate",
                params={"session_id": "leak-test-invalidate", "decision_label": "postgresql"},
            )
        assert r.status_code == 500
        assert "hunter2" not in r.text
