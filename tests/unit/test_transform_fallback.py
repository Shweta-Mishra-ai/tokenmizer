"""
A rejected transformed request must not end the session.

The request the provider sees is the one TokenMizer built — windowed,
compressed, with context injected. When that is what fails, the fault may
be in a transform rather than in the provider, and before this a windowing
bug that produced an assistant-first message list turned into a 502 on
every remaining turn of every long session.

Rule: if the transformed request fails and differs from what the client
sent, retry once with the client's own (redacted) messages. Success means
the turn goes through at full price, savings for the turn are zero, the
response says so, and the event is counted so it cannot pass unnoticed. A
second failure is the 502 it always was.
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

# Enough history that windowing and compression actually change the
# request; the transformed list must differ from the raw one.
HISTORY = []
for i in range(30):
    HISTORY.append({"role": "user", "content": f"user turn {i} " + "word " * 80})
    HISTORY.append({"role": "assistant", "content": f"assistant turn {i} " + "word " * 80})
HISTORY.append({"role": "user", "content": "what did we decide about the database?"})


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module.settings.cache, "enabled", False)
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
    monkeypatch.setattr(app_module.settings.memory, "max_tokens_before_summary", 1000)
    with TestClient(app) as c:
        yield c


def _post(client, session_id):
    return client.post("/v1/chat/completions", json={
        "model": "gpt-4o", "session_id": session_id, "messages": HISTORY})


def test_transformed_failure_is_retried_with_the_clients_messages(client, monkeypatch):
    seen = []

    async def chat(messages, **kw):
        seen.append(messages)
        if len(seen) == 1:
            raise RuntimeError("400 first message must use the 'user' role")
        return OK

    provider = AsyncMock()
    provider.chat = chat
    monkeypatch.setattr(app_module, "_get_provider", lambda: provider)
    failures_before = app_module._analytics.persist_failures.get("transform_rejected", 0)

    r = _post(client, "fallback-ok")

    assert r.status_code == 200, r.text
    assert len(seen) == 2, "expected exactly one retry"
    assert seen[0] != seen[1], "the retry must not resend the same request"
    # The retry is the client's own messages, redaction aside.
    assert [m["content"] for m in seen[1]] == [m["content"] for m in HISTORY]

    body = r.json()
    assert body["tokenmizer"]["fallback"]["transform_rejected"] is True
    assert body["tokenmizer"]["fallback"]["ref"]
    assert body["tokenmizer"]["total_saved"] == 0, "no savings on a fallback turn"
    assert body["usage"]["tokens_saved"] == 0
    assert app_module._analytics.persist_failures["transform_rejected"] == failures_before + 1


def test_second_failure_is_still_a_502(client, monkeypatch):
    calls = []

    async def chat(messages, **kw):
        calls.append(1)
        raise RuntimeError("provider is down")

    provider = AsyncMock()
    provider.chat = chat
    monkeypatch.setattr(app_module, "_get_provider", lambda: provider)

    r = _post(client, "fallback-fails")
    assert r.status_code == 502
    assert len(calls) == 2
    assert "ref:" in r.json()["detail"]
    assert "provider is down" not in r.text, "exception text must not reach the client"


def test_no_retry_when_nothing_was_transformed(client, monkeypatch):
    """A short request that no layer changed has nothing to fall back to;
    retrying it would just double the bill on a genuine provider outage."""
    calls = []

    async def chat(messages, **kw):
        calls.append(1)
        raise RuntimeError("provider is down")

    provider = AsyncMock()
    provider.chat = chat
    monkeypatch.setattr(app_module, "_get_provider", lambda: provider)
    monkeypatch.setattr(app_module.settings.terse_output, "enabled", False)
    monkeypatch.setattr(app_module.settings.graph_checkpoint, "enabled", False)

    r = client.post("/v1/chat/completions", json={
        "model": "gpt-4o", "session_id": "no-transform",
        "messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == 502
    assert len(calls) == 1


def test_a_healthy_request_makes_exactly_one_call(client, monkeypatch):
    calls = []

    async def chat(messages, **kw):
        calls.append(1)
        return OK

    provider = AsyncMock()
    provider.chat = chat
    monkeypatch.setattr(app_module, "_get_provider", lambda: provider)

    r = _post(client, "healthy")
    assert r.status_code == 200
    assert len(calls) == 1
    assert "fallback" not in r.json()["tokenmizer"]
