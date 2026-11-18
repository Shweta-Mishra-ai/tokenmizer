"""
/health must not report "ok" while memory is being lost.

It returned the literal string "ok" whatever had happened, so a
deployment whose checkpoint writes had been failing for a day, or whose
graph database had been quarantined after corruption, looked healthy to
every uptime monitor pointed at it. The counters existed — they were just
only reachable from /api/stats and the server log.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tokenmizer.api import app as app_module
from tokenmizer.api.app import app
from tokenmizer.graph_memory.graph import GraphMemory


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
    app_module._graph_cache.clear()
    # The analytics engine is a module singleton shared by the whole suite,
    # and /health now reads its counters — so this file must leave them as
    # it found them or every later test that asserts a healthy status
    # depends on run order.
    saved = dict(app_module._analytics._persist_failures)
    app_module._analytics._persist_failures.clear()
    monkeypatch.setattr(app_module._checkpoint_mgr, "persistence_broken", False)
    monkeypatch.setattr(app_module._checkpoint_mgr, "data_loss_detected", False)
    try:
        with TestClient(app) as c:
            yield c, tmp_path
    finally:
        app_module._graph_cache.clear()
        app_module._analytics._persist_failures.clear()
        app_module._analytics._persist_failures.update(saved)


def test_healthy_when_nothing_has_failed(client):
    c, _ = client
    body = c.get("/health").json()
    assert body["status"] == "ok"
    assert body["persist_failures"] == {}
    assert body["version"]


def test_a_failed_write_degrades_the_status(client):
    c, _ = client
    app_module._analytics.record_silent_failure("checkpoint")
    body = c.get("/health").json()
    assert body["status"] == "degraded"
    assert body["persist_failures"]["checkpoint"] == 1


def test_an_unreadable_graph_degrades_the_status(client, tmp_path):
    c, _ = client
    g = GraphMemory("unreadable", storage_dir=str(tmp_path))
    g._load_failed = True
    app_module._graph_cache["unreadable"] = g
    body = c.get("/health").json()
    assert body["status"] == "degraded"
    assert body["sessions_with_unreadable_graph"] == ["unreadable"]


def test_storage_without_durability_degrades_the_status(client, tmp_path):
    c, _ = client
    g = GraphMemory("no-durable", storage_dir=str(tmp_path))
    g._persistence_broken = True
    app_module._graph_cache["no-durable"] = g
    body = c.get("/health").json()
    assert body["status"] == "degraded"
    assert body["sessions_without_durable_storage"] == ["no-durable"]


def test_broken_checkpoint_storage_degrades_the_status(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(app_module._checkpoint_mgr, "persistence_broken", True)
    body = c.get("/health").json()
    assert body["status"] == "degraded"
    assert body["checkpoint_storage_broken"] is True


def test_health_needs_no_api_key(client, monkeypatch):
    """An uptime monitor has no credential; the endpoint must stay open."""
    c, _ = client
    monkeypatch.setattr(app_module.settings, "api_key", "secret", raising=False)
    assert c.get("/health").status_code == 200
