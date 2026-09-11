"""
GET /api/sessions — the caller's sessions, and only the caller's.

The dashboard used to render a hard-coded legend labelled "(demo)" in the
place a session list belongs, because nothing could tell it which sessions
existed. The interactive graph viewer at /api/graph/{id}/html was reachable
only by typing the URL.

The list is scoped by ownership. graph_memory.db holds every principal's
sessions in one file; listing from it would show one API key's sessions to
another, so the ownership store is the source and that is what these tests
check first.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from tokenmizer.api import app as app_module
from tokenmizer.api.app import app
from tokenmizer.providers.providers import LLMResponse


@pytest.fixture
def client(monkeypatch):
    provider = AsyncMock()
    provider.chat = AsyncMock(return_value=LLMResponse(
        text="ok", input_tokens=10, output_tokens=5,
        model="fake-model", provider="fake", latency_ms=1.0,
    ))
    monkeypatch.setattr(app_module, "_get_provider", lambda: provider)
    monkeypatch.setattr(app_module.settings.cache, "enabled", False)
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
    with TestClient(app) as c:
        yield c


def _chat(client, session_id):
    return client.post("/v1/chat/completions", json={
        "model": "gpt-4o", "session_id": session_id,
        "messages": [
            {"role": "user", "content": "we need a datastore for the checkout service"},
            {"role": "assistant", "content": "Decided: PostgreSQL for order storage."},
            {"role": "user", "content": "which database did we choose for orders"},
        ],
    })


def test_lists_the_callers_sessions_with_detail(client):
    assert _chat(client, "sessions-a").status_code == 200
    r = client.get("/api/sessions")
    assert r.status_code == 200, r.text
    ids = {s["session_id"] for s in r.json()["sessions"]}
    assert "sessions-a" in ids
    row = next(s for s in r.json()["sessions"] if s["session_id"] == "sessions-a")
    assert row["node_count"] >= 1
    assert row["graph_url"] == "/api/graph/sessions-a/html"
    assert "by_type" in row


def test_another_principals_sessions_are_not_listed(client, monkeypatch):
    """The whole reason the list comes from the ownership store."""
    _chat(client, "sessions-mine")
    app_module._ownership.claim("sessions-theirs", "someone-else")

    ids = {s["session_id"] for s in client.get("/api/sessions").json()["sessions"]}
    assert "sessions-mine" in ids
    assert "sessions-theirs" not in ids


def test_empty_list_is_a_200_not_an_error(client, monkeypatch):
    monkeypatch.setattr(app_module._ownership, "sessions_for", lambda p: [])
    r = client.get("/api/sessions")
    assert r.status_code == 200
    assert r.json() == {"sessions": [], "count": 0}


def test_dashboard_links_to_the_real_viewer(client):
    """The dashboard must fetch /api/sessions and point at the interactive
    viewer, not render a static legend."""
    html = client.get("/").text
    assert "/api/sessions" in html
    assert "/api/graph/" in html and "/html" in html
    assert "(demo)" not in html
