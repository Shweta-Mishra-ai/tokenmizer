"""
POST /api/checkpoint accepts a transcript.

The route claimed the session (so it showed up in the dashboard's list)
and called CheckpointManager.create(messages=[]), so a checkpoint made
through the MCP tool or the CLI produced a listed session with zero
nodes — the most common way a user first met an empty graph page.
create() already extracts from whatever messages it is given; the route
just never passed any.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tokenmizer.api import app as app_module
from tokenmizer.api.app import app
from tokenmizer.security.ownership import OwnershipStore

TRANSCRIPT = [
    {"role": "user", "content": "We're building the checkout service."},
    {"role": "assistant", "content":
     "Decided: use PostgreSQL for order storage. Created internal/store/postgres.go."},
]


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
    monkeypatch.setattr(app_module.settings.graph_checkpoint, "storage_dir", str(tmp_path))
    monkeypatch.setattr(app_module, "_ownership", OwnershipStore(storage_dir=str(tmp_path)))
    app_module._graph_cache.clear()
    with TestClient(app) as c:
        yield c


def test_transcript_in_the_body_is_extracted_into_the_graph(client):
    r = client.post("/api/checkpoint?session_id=ckpt-msgs", json={"messages": TRANSCRIPT})
    assert r.status_code == 200, r.text
    assert r.json()["node_count"] >= 1

    stats = client.get("/api/graph/ckpt-msgs").json()
    assert stats["node_count"] >= 1
    assert "decision" in stats["by_type"]


def test_transcript_is_redacted_before_it_reaches_the_graph(client):
    r = client.post("/api/checkpoint?session_id=ckpt-redact", json={"messages": [
        {"role": "user", "content": "We're building the checkout service."},
        {"role": "assistant", "content":
         "Decided: use PostgreSQL for orders, key sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz"},
    ]})
    assert r.status_code == 200, r.text
    viz = client.get("/api/graph/ckpt-redact/viz").json()
    dumped = str(viz)
    assert "sk-ant-api03-abcdefghij" not in dumped


def test_no_body_keeps_the_old_behaviour(client):
    r = client.post("/api/checkpoint?session_id=ckpt-empty")
    assert r.status_code == 200, r.text
    assert r.json()["node_count"] == 0


def test_malformed_messages_is_a_422(client):
    r = client.post("/api/checkpoint?session_id=ckpt-bad", json={"messages": "nope"})
    assert r.status_code == 422
