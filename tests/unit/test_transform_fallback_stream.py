"""
Streaming counterpart to test_transform_fallback.py.

Same rule, adapted to SSE: if the transformed request fails on the
provider's first chunk — before anything has been streamed to the client
— retry once with the client's own (redacted) messages. Success means the
turn goes through at full price, savings for the turn are zero, and the
event is counted. A failure AFTER partial content has already streamed to
the client can never retry (that would duplicate or interleave output),
so it is always the existing mid-stream error event, unchanged.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tokenmizer.api import app as app_module
from tokenmizer.api.app import app
from tokenmizer.providers.providers import ProviderError

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


def _post_stream(client, session_id, messages=None):
    return client.post("/v1/chat/completions", json={
        "model": "gpt-4o", "session_id": session_id,
        "messages": messages if messages is not None else HISTORY,
        "stream": True,
    })


def _events(response_text: str) -> list[dict]:
    events = []
    for line in response_text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[len("data: "):]))
    return events


def test_first_chunk_failure_is_retried_with_the_clients_messages(client, monkeypatch):
    seen = []

    class Provider:
        async def chat_stream(self, messages, **kw):
            seen.append(messages)
            if len(seen) == 1:
                raise ProviderError("openai", "bad_request",
                                    "400 first message must use the 'user' role")
                yield  # pragma: no cover — makes this an async generator function
            yield "answer: "
            yield "postgres"

    monkeypatch.setattr(app_module, "_get_provider", lambda: Provider())
    failures_before = app_module._analytics.persist_failures.get("transform_rejected", 0)

    r = _post_stream(client, "stream-fallback-ok")

    assert r.status_code == 200, r.text
    assert len(seen) == 2, "expected exactly one retry"
    assert seen[0] != seen[1], "the retry must not resend the same request"
    assert [m["content"] for m in seen[1]] == [m["content"] for m in HISTORY]

    events = _events(r.text)
    full_text = "".join(
        e["choices"][0]["delta"].get("content", "") for e in events
    )
    assert full_text == "answer: postgres"
    assert not any("error" in e for e in events), "no error event on a successful fallback"

    final = [e for e in events if e["choices"][0]["finish_reason"] == "stop"][0]
    assert final["tokenmizer"]["fallback"]["transform_rejected"] is True
    assert app_module._analytics.persist_failures["transform_rejected"] == failures_before + 1


def test_second_failure_is_still_an_error_event(client, monkeypatch):
    calls = []

    class Provider:
        async def chat_stream(self, messages, **kw):
            calls.append(1)
            raise ProviderError("openai", "down", "provider is down")
            yield  # pragma: no cover

    monkeypatch.setattr(app_module, "_get_provider", lambda: Provider())

    r = _post_stream(client, "stream-fallback-fails")

    assert r.status_code == 200
    assert len(calls) == 2, "expected the retry, then giving up"
    events = _events(r.text)
    assert any(e.get("error", {}).get("type") == "provider_error" for e in events)


def test_no_retry_once_partial_content_has_streamed(client, monkeypatch):
    """A failure after real content already reached the client can never
    retry — that would duplicate or interleave output — so this must stay
    exactly the existing single error event, with no second call."""
    calls = []

    class Provider:
        async def chat_stream(self, messages, **kw):
            calls.append(1)
            yield "partial answer"
            raise ProviderError("openai", "overloaded", "upstream died mid-stream")

    monkeypatch.setattr(app_module, "_get_provider", lambda: Provider())

    r = _post_stream(client, "stream-fallback-partial")

    assert r.status_code == 200
    assert len(calls) == 1, "must not retry once content has already streamed"
    events = _events(r.text)
    assert any(e.get("error", {}).get("type") == "provider_error" for e in events)


def test_no_retry_when_nothing_was_transformed(client, monkeypatch):
    """A short request no layer changed has nothing to fall back to."""
    calls = []

    class Provider:
        async def chat_stream(self, messages, **kw):
            calls.append(1)
            raise ProviderError("openai", "down", "provider is down")
            yield  # pragma: no cover

    monkeypatch.setattr(app_module, "_get_provider", lambda: Provider())
    monkeypatch.setattr(app_module.settings.terse_output, "enabled", False)
    monkeypatch.setattr(app_module.settings.graph_checkpoint, "enabled", False)

    r = _post_stream(client, "stream-no-transform",
                     messages=[{"role": "user", "content": "hello"}])

    assert r.status_code == 200
    assert len(calls) == 1


def test_a_healthy_stream_makes_exactly_one_call(client, monkeypatch):
    calls = []

    class Provider:
        async def chat_stream(self, messages, **kw):
            calls.append(1)
            yield "ok"

    monkeypatch.setattr(app_module, "_get_provider", lambda: Provider())

    r = _post_stream(client, "stream-healthy")

    assert r.status_code == 200
    assert len(calls) == 1
    events = _events(r.text)
    final = [e for e in events if e["choices"][0]["finish_reason"] == "stop"][0]
    assert "tokenmizer" not in final
