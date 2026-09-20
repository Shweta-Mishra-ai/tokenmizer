"""
Short follow-ups must still get session memory, and the graph must be
mutated under one lock.

Two defects on the same code path:

1. Context injection was gated on `len(user_query.split()) >= 4`, so "why?",
   "continue", "fix it", "same as before" and "run the tests" received no
   graph context at all. Those are precisely the turns where the model most
   needs to be told what "it" and "before" refer to: the graph held the
   answer and the proxy declined to pass it on, on a word count.

2. _background_extract runs under _get_session_lock and mutates the graph,
   but the foreground request path took no lock, so turn N's background
   extraction could interleave with turn N+1's foreground mutation.
"""
from __future__ import annotations

import asyncio
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
        yield c, provider


def _sent_system_text(provider) -> str:
    messages = provider.chat.call_args.kwargs["messages"]
    return "\n".join(m["content"] for m in messages if m.get("role") == "system")


# A session with enough substance for the graph to have something to say.
HISTORY = [
    {"role": "user", "content":
     "We are building the checkout service and need to pick a datastore."},
    {"role": "assistant", "content":
     "Decided: PostgreSQL for order storage. Created internal/store/postgres.go."},
    {"role": "user", "content":
     "Add retries with exponential backoff to the payment client please."},
    {"role": "assistant", "content":
     "Working on exponential backoff in internal/pay/client.go."},
]


@pytest.mark.parametrize("followup", ["why?", "continue", "fix it", "keep going"])
def test_short_followup_still_receives_session_context(client, followup):
    c, provider = client
    c.post("/v1/chat/completions", json={
        "model": "gpt-4o", "session_id": f"short-{followup.replace('?', '')}",
        "messages": HISTORY + [{"role": "user", "content": followup}],
    })
    assert "relevant session context" in _sent_system_text(provider), (
        f"a {len(followup.split())}-word follow-up got no graph context; "
        "these are the turns that most need it"
    )


def test_long_query_behaviour_is_unchanged(client):
    c, provider = client
    c.post("/v1/chat/completions", json={
        "model": "gpt-4o", "session_id": "long-query",
        "messages": HISTORY + [
            {"role": "user", "content": "which datastore did we choose for orders"}],
    })
    assert "relevant session context" in _sent_system_text(provider)


def test_last_substantive_query_walks_back_past_short_turns():
    from tokenmizer.api.app import _last_substantive_query

    assert _last_substantive_query([
        {"role": "user", "content": "how should we store the order records"},
        {"role": "assistant", "content": "PostgreSQL."},
        {"role": "user", "content": "why?"},
    ]) == "how should we store the order records"


def test_last_substantive_query_ignores_assistant_turns():
    from tokenmizer.api.app import _last_substantive_query

    assert _last_substantive_query([
        {"role": "assistant", "content": "a long assistant explanation of things"},
        {"role": "user", "content": "ok"},
    ]) == ""


# ── The lock ─────────────────────────────────────────────────────────────────

def test_graph_mutation_holds_the_session_lock(client, monkeypatch):
    """Assert the foreground actually holds the lock the background
    extractor also takes — not merely that a lock object exists."""
    held: list[bool] = []
    original = app_module._update_graph

    async def spy(session_id, *args, **kwargs):
        held.append(app_module._get_session_lock(session_id).locked())
        return await original(session_id, *args, **kwargs)

    monkeypatch.setattr(app_module, "_update_graph", spy)

    c, _ = client
    c.post("/v1/chat/completions", json={
        "model": "gpt-4o", "session_id": "lock-check",
        "messages": HISTORY + [{"role": "user", "content": "why?"}],
    })
    assert held == [True], f"session lock not held during graph mutation: {held}"


def test_foreground_lock_does_not_deadlock_the_background_extractor():
    """The foreground now holds the same lock _background_extract waits on.
    That must serialise them, not wedge them: the foreground never awaits the
    background task, so the task simply runs after the request lets go."""
    async def scenario():
        lock = app_module._get_session_lock("deadlock-check")
        ran = asyncio.Event()

        async def background():
            async with lock:
                ran.set()

        async with lock:
            task = asyncio.create_task(background())
            await asyncio.sleep(0)
            assert not ran.is_set(), "background ran while foreground held the lock"

        await asyncio.wait_for(task, timeout=2.0)
        assert ran.is_set()

    asyncio.run(scenario())
