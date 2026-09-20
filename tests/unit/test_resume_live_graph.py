"""
GET /api/resume must answer from the graph, not only from the checkpoint
table.

The auto-checkpoint trigger measures the request AFTER windowing, and
windowing keeps the request small, so on a default config a long proxy
session can run for hours without ever crossing the threshold. The graph
is persisted on every turn regardless. "No checkpoint found" was the
answer to a session full of memory; a checkpoint older than the graph
returned a snapshot missing every decision made since.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tokenmizer.api import app as app_module
from tokenmizer.api.app import app
from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
    monkeypatch.setattr(app_module.settings.graph_checkpoint, "storage_dir", str(tmp_path))
    monkeypatch.setattr(app_module._checkpoint_mgr, "_db_path", tmp_path / "checkpoints.db")
    app_module._checkpoint_mgr._init_db()
    app_module._graph_cache.clear()
    with TestClient(app) as c:
        yield c, tmp_path


def _seed(session_id, storage_dir):
    g = GraphMemory(session_id, storage_dir=str(storage_dir))
    g.add_node(NodeType.GOAL, "Build the checkout service", NodeStatus.IN_PROGRESS,
               importance=1.0)
    g.add_node(NodeType.DECISION, "Use PostgreSQL for order storage",
               NodeStatus.COMPLETED, importance=0.9)
    app_module._graph_cache[session_id] = g
    return g


def test_no_checkpoint_but_a_graph_resumes_from_the_live_graph(client):
    c, storage = client
    _seed("live-only", storage)
    r = c.get("/api/resume/live-only")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "live_graph"
    assert body["checkpoint_id"] is None
    assert "PostgreSQL" in body["resume_context"]
    assert body["token_count"] > 0


@pytest.mark.parametrize("level", ["critical", "standard", "full"])
def test_every_level_works_from_the_live_graph(client, level):
    c, storage = client
    _seed(f"live-{level}", storage)
    r = c.get(f"/api/resume/live-{level}", params={"level": level})
    assert r.status_code == 200, r.text
    assert "PostgreSQL" in r.json()["resume_context"]


def test_no_checkpoint_and_empty_graph_is_still_a_404(client):
    c, _ = client
    r = c.get("/api/resume/never-seen")
    assert r.status_code == 404
    assert "No checkpoint found" in r.json()["detail"]


def test_checkpoint_newer_than_graph_is_served_as_is(client):
    c, storage = client
    g = _seed("ckpt-fresh", storage)
    ckpt = app_module._checkpoint_mgr.create(
        session_id="ckpt-fresh", messages=[], graph=g, context_pct=0.5, trigger="manual")
    r = c.get("/api/resume/ckpt-fresh")
    body = r.json()
    assert body["source"] == "checkpoint"
    assert body["checkpoint_id"] == ckpt.checkpoint_id


def test_graph_updated_after_checkpoint_wins_but_keeps_the_continue_hint(client):
    c, storage = client
    g = _seed("ckpt-stale", storage)
    ckpt = app_module._checkpoint_mgr.create(
        session_id="ckpt-stale",
        messages=[{"role": "user", "content": "now add the refund endpoint"}],
        graph=g, context_pct=0.5, trigger="manual")
    # Pretend the checkpoint is a day old, then the session went on.
    conn = app_module._checkpoint_mgr._db_connect()
    conn.execute("UPDATE checkpoints SET created_at = created_at - 86400 "
                 "WHERE checkpoint_id=?", (ckpt.checkpoint_id,))
    conn.commit()
    conn.close()
    g.add_node(NodeType.DECISION, "Use Stripe for refunds", NodeStatus.COMPLETED,
               importance=0.9)

    r = c.get("/api/resume/ckpt-stale")
    body = r.json()
    assert body["source"] == "live_graph"
    assert body["checkpoint_id"] == ckpt.checkpoint_id
    assert "Stripe" in body["resume_context"], "the decision made after the checkpoint is missing"
    assert "refund endpoint" in body["resume_context"], "the checkpoint's continue hint was dropped"
