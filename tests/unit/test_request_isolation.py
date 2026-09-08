"""
The pipeline must not mutate the caller's messages.

`messages = raw_messages[:]` copies the list but shares every dict, so
Layer 2's terse-prompt injection —
`m["content"] = terse + "\n\n" + m["content"]` — rewrote a dict that
raw_messages still pointed at.

raw_messages is the *baseline*. It feeds:
  - orig_input_tokens, the denominator of tokens_saved and of every savings
    percentage reported to the client, /api/stats, and the dashboard
  - graph.extract_from_messages(), so the injected prompt became a candidate
    for extraction into the graph as if it were session content
  - CheckpointManager.create(), so it survived into resume

The user-visible result was systematically overstated savings: TokenMizer
counted its own injected prompt as part of the request the user sent.
These tests assert the reported baseline matches what the client actually
posted.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from tokenmizer.api import app as app_module
from tokenmizer.api.app import app
from tokenmizer.core.tokenizer import count_messages_tokens
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
    # The layer that did the mutating — on, so the regression is reachable.
    monkeypatch.setattr(app_module.settings.terse_output, "enabled", True)
    with TestClient(app) as c:
        yield c


SENT = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "explain database indexing please"},
]


def test_reported_original_tokens_match_what_the_client_sent(client):
    """original_prompt_tokens is the savings denominator — it must be the
    caller's request, not the caller's request plus TokenMizer's prompt."""
    r = client.post("/v1/chat/completions",
                    json={"model": "gpt-4o", "messages": SENT,
                          "session_id": "isolation-baseline"})
    assert r.status_code == 200, r.text

    reported = r.json()["usage"]["original_prompt_tokens"]
    expected = count_messages_tokens(SENT, "gpt-4o")

    assert reported == expected, (
        f"reported baseline {reported} != what the client sent {expected}. "
        "The terse-output prompt leaked into raw_messages, inflating the "
        "denominator of every savings figure."
    )


def test_terse_prompt_is_not_counted_as_the_users_own_input(client):
    """A direct read of the same bug: the injected text must not appear in
    the baseline, however the count is derived."""
    terse = app_module._compression.terse_system_prompt(
        app_module.settings.terse_output.level
    )
    marker = terse.split(".")[0][:40]

    r = client.post("/v1/chat/completions",
                    json={"model": "gpt-4o", "messages": SENT,
                          "session_id": "isolation-marker"})
    assert r.status_code == 200, r.text

    baseline = count_messages_tokens(SENT, "gpt-4o")
    with_terse = count_messages_tokens(
        [{"role": "system", "content": terse + "\n\n" + SENT[0]["content"]},
         SENT[1]], "gpt-4o",
    )
    assert with_terse > baseline, (
        f"test is not exercising the bug — terse prompt {marker!r} is empty "
        "or free"
    )
    assert r.json()["usage"]["original_prompt_tokens"] != with_terse


def test_caller_message_dicts_are_not_mutated(client):
    """The request objects FastAPI built are per-request, but the same
    aliasing would corrupt any caller that reused its list — and it is the
    property the fix actually establishes."""
    payload_messages = [dict(m) for m in SENT]
    before = [dict(m) for m in payload_messages]

    r = client.post("/v1/chat/completions",
                    json={"model": "gpt-4o", "messages": payload_messages,
                          "session_id": "isolation-mutation"})
    assert r.status_code == 200, r.text
    assert payload_messages == before
