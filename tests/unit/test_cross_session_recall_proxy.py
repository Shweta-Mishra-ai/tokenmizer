"""
Cross-session recall wired into the proxy's context-injection step
(api/app.py::_update_graph). Off by default; this exercises the path with
Settings.graph_checkpoint.cross_session_recall explicitly enabled.

Mirrors tests/unit/test_transform_fallback.py's TestClient pattern: a real
HTTP round trip through chat_completions() with a fake provider, so the
assertion is on what the proxy actually sent the model rather than on an
internal function signature.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from tokenmizer.api import app as app_module
from tokenmizer.api.app import app
from tokenmizer.providers.providers import LLMResponse

OK = LLMResponse(text="ok", input_tokens=10, output_tokens=5,
                 model="fake", provider="fake", latency_ms=1.0)

# Enough turns and enough distinct signal (goal, decision, task) that
# heuristic extraction gives this session's OWN graph >= 3 nodes on its
# own — the injection gate at _update_graph requires that regardless of
# cross-session recall, so a same-session-only fixture would never reach
# the code path under test.
SESSION_B_HISTORY = [
    {"role": "user", "content": "We're building the billing service."},
    {"role": "assistant", "content": "Goal: ship the billing service."},
    {"role": "assistant", "content": "Working on: the invoice PDF export."},
    {"role": "assistant", "content":
     "Decided: use Stripe for payment processing. Created internal/pay/stripe.go."},
    {"role": "user", "content": "What did we decide about the database?"},
]


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.settings.cache, "enabled", False)
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
    monkeypatch.setattr(app_module.settings.graph_checkpoint, "storage_dir", str(tmp_path))
    monkeypatch.setattr(app_module.settings.graph_checkpoint, "cross_session_recall", True)
    from tokenmizer.security.ownership import OwnershipStore
    monkeypatch.setattr(app_module, "_ownership", OwnershipStore(storage_dir=str(tmp_path)))
    app_module._graph_cache.clear()
    with TestClient(app) as c:
        yield c


def _chat(client, session_id, messages, provider):
    def _get(): return provider
    return client.post("/v1/chat/completions", json={
        "model": "gpt-4o", "session_id": session_id, "messages": messages,
    })


def test_a_second_session_recalls_the_first_sessions_decision(client, monkeypatch):
    seen = []

    async def chat(messages, **kw):
        seen.append(messages)
        return OK

    provider = AsyncMock()
    provider.chat = chat
    monkeypatch.setattr(app_module, "_get_provider", lambda: provider)

    r1 = client.post("/v1/chat/completions", json={
        "model": "gpt-4o", "session_id": "proxy-a", "messages": [
            {"role": "user", "content": "We're building the checkout service."},
            {"role": "assistant", "content":
             "Decided: use PostgreSQL for order storage. "
             "Created internal/store/postgres.go."},
        ],
    })
    assert r1.status_code == 200, r1.text

    r2 = client.post("/v1/chat/completions", json={
        "model": "gpt-4o", "session_id": "proxy-b", "messages": SESSION_B_HISTORY,
    })
    assert r2.status_code == 200, r2.text

    sent_to_model = seen[-1]
    joined = " ".join(m["content"] for m in sent_to_model if isinstance(m.get("content"), str))
    assert "postgresql" in joined.lower(), (
        "session proxy-b should have recalled proxy-a's decision once "
        "cross_session_recall is on"
    )


def test_off_by_default_does_not_leak_across_sessions(client, monkeypatch):
    monkeypatch.setattr(app_module.settings.graph_checkpoint, "cross_session_recall", False)
    seen = []

    async def chat(messages, **kw):
        seen.append(messages)
        return OK

    provider = AsyncMock()
    provider.chat = chat
    monkeypatch.setattr(app_module, "_get_provider", lambda: provider)

    client.post("/v1/chat/completions", json={
        "model": "gpt-4o", "session_id": "proxy-a2", "messages": [
            {"role": "user", "content": "We're building the checkout service."},
            {"role": "assistant", "content":
             "Decided: use PostgreSQL for order storage. "
             "Created internal/store/postgres.go."},
        ],
    })
    client.post("/v1/chat/completions", json={
        "model": "gpt-4o", "session_id": "proxy-b2", "messages": SESSION_B_HISTORY,
    })

    sent_to_model = seen[-1]
    joined = " ".join(m["content"] for m in sent_to_model if isinstance(m.get("content"), str))
    assert "postgresql" not in joined.lower()
