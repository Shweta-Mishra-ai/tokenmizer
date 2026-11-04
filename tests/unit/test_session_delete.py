"""
DELETE /api/sessions/{id}: there was no way to forget a session short of
deleting SQLite files by hand, though a session holds whatever was pasted
into it.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from tokenmizer.api import app as app_module
from tokenmizer.providers.providers import LLMResponse


@pytest.fixture
def client(monkeypatch):
    fake = AsyncMock()
    fake.chat = AsyncMock(return_value=LLMResponse(
        text="Noted.", input_tokens=10, output_tokens=2, model="fake", provider="fake"))
    monkeypatch.setattr(app_module, "_get_provider", lambda: fake)
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
    monkeypatch.setattr(app_module.settings, "api_keys", [], raising=False)
    with TestClient(app_module.app) as c:
        yield c


def _chat(c, sid, text):
    r = c.post("/v1/chat/completions", json={
        "model": "m", "session_id": sid, "messages": [{"role": "user", "content": text}]})
    assert r.status_code == 200, r.text


def test_delete_removes_memory_checkpoints_and_ownership(client):
    sid = f"del-{uuid.uuid4().hex[:8]}"
    _chat(client, sid, "Decided: use PostgreSQL because of JSONB. My token is hunter2-secret.")
    assert client.post("/api/checkpoint", params={"session_id": sid}).status_code == 200
    assert "PostgreSQL" in client.get(f"/api/resume/{sid}").text

    r = client.delete(f"/api/sessions/{sid}")
    assert r.status_code == 200, r.text
    gone = r.json()["deleted"]
    assert gone["graph_rows"] > 0 and gone["checkpoints"] >= 1

    assert sid not in app_module._graph_cache
    assert app_module._ownership.owner_of(sid) is None
    assert "PostgreSQL" not in client.get(f"/api/resume/{sid}").text
    assert app_module._checkpoint_mgr.delete_session(sid) == 0


def test_only_the_owner_may_delete(client, monkeypatch):
    sid = f"del-{uuid.uuid4().hex[:8]}"
    app_module._ownership.claim(sid, "k_someone_else")
    r = client.delete(f"/api/sessions/{sid}")
    assert r.status_code == 404
    assert app_module._ownership.owner_of(sid) == "k_someone_else"


def test_a_graph_object_older_than_the_delete_cannot_write_it_back(tmp_path):
    """A background extraction captures the graph and waits for the session
    lock; if the delete wins the lock, the extraction then persisted the
    session straight back. Objects created before the delete are orphans."""
    from tokenmizer.graph_memory.graph import GraphMemory
    from tokenmizer.graph_memory.persistence import delete_session
    from tokenmizer.graph_memory.types import NodeType

    sid = f"orphan-{uuid.uuid4().hex[:8]}"
    g = GraphMemory(sid, storage_dir=str(tmp_path))
    g.add_node(NodeType.DECISION, "Use PostgreSQL for storage")
    assert g._persist()
    delete_session(str(tmp_path), sid)

    g.add_node(NodeType.DECISION, "Use Redis for caching")   # late background write
    g._persist()
    assert len(GraphMemory(sid, storage_dir=str(tmp_path))._nodes) == 0

    fresh = GraphMemory(sid, storage_dir=str(tmp_path))      # the id reused afterwards
    fresh.add_node(NodeType.DECISION, "Use SQLite for tests")
    assert fresh._persist()
    assert len(GraphMemory(sid, storage_dir=str(tmp_path))._nodes) == 1


def test_cached_answers_of_the_session_are_dropped(client):
    sid = f"del-{uuid.uuid4().hex[:8]}"
    cache = app_module._cache
    cache.set("what is our db", "PostgreSQL", input_tokens=1, output_tokens=1, session_id=sid)
    assert client.delete(f"/api/sessions/{sid}").json()["deleted"]["cached_answers"] >= 1
    assert not any(e.scope == sid for e in cache._exact.values())
