"""
Regression tests for three related bugs in chat_completions()'s analytics
and error-handling paths (TM-11, TM-33, TM-34).

TM-11: input_tokens_sent (the "tokens actually sent" figure reported in
analytics and the dashboard) was measured on `messages` BEFORE
_update_graph() applied windowing and graph-context injection — so it
never reflected either the reduction from windowing or the addition from
context injection. Fixed by measuring it right before the provider call,
against the final payload.

TM-33: provider exceptions were echoed verbatim into the 502 response
body. Provider SDK errors routinely include request URLs, header
fragments, or other details that shouldn't reach an API client. Fixed by
logging full detail server-side and returning a generic message with a
correlation id.

TM-34: cache_hit was inferred as `input_tokens_actual == 0 and
response_text != ""` — a provider that legitimately reports zero input
tokens (or a hypothetical future provider quirk) would be misrecorded as
a cache hit. Fixed by returning the flag explicitly from _call_provider
instead of inferring it downstream.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from tokenmizer.api import app as app_module
from tokenmizer.api.app import app
from tokenmizer.providers.providers import LLMResponse


def _fake_response(text: str = "hello") -> LLMResponse:
    return LLMResponse(text=text, input_tokens=10, output_tokens=5,
                       model="fake", provider="fake", latency_ms=1.0)


@pytest.fixture
def client(monkeypatch):
    fake_provider = AsyncMock()
    fake_provider.chat = AsyncMock(return_value=_fake_response())
    monkeypatch.setattr(app_module, "_get_provider", lambda: fake_provider)
    monkeypatch.setattr(app_module.settings.cache, "enabled", False)
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
    with TestClient(app) as c:
        yield c, fake_provider


class TestSentTokensReflectFinalPayload:

    def test_sent_tokens_reflect_windowing_reduction(self, client, monkeypatch):
        """When windowing fires and shrinks the payload, input_tokens_sent
        must reflect the SMALLER post-windowing size, not the original."""
        c, fake = client
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "enabled", True)
        monkeypatch.setattr(app_module.settings.memory, "max_tokens_before_summary", 30)
        monkeypatch.setattr(app_module.settings.memory, "recent_turns_verbatim", 1)
        # _smart_window is a singleton constructed once at import time with
        # its own captured token_budget/protect_recent — mutating settings
        # above doesn't propagate to it (matches how existing window tests
        # construct their own SmartMessageWindow instance directly rather
        # than relying on settings mutation). Patch its actual gating
        # attributes too, or .apply() compares against the stale default
        # (4000) and never windows regardless of the settings change above.
        monkeypatch.setattr(app_module._smart_window, "token_budget", 30)
        monkeypatch.setattr(app_module._smart_window, "protect_recent", 1)

        recorded = {}
        monkeypatch.setattr(
            app_module._analytics, "record",
            lambda **kwargs: recorded.update(kwargs),
        )

        big_messages = [
            {"role": "user", "content": " ".join([f"word{i}"] * 15)}
            for i in range(10)
        ]
        r = c.post("/v1/chat/completions", json={
            "messages": big_messages, "session_id": "sent-tokens-test",
        })
        assert r.status_code == 200, r.text
        assert recorded, "analytics.record was never called"

        # The pre-windowing size would be much larger than what's
        # actually left after windowing collapses old turns into a
        # single bridge message; input_tokens_sent must track the latter.
        original_size = recorded["input_tokens_original"]
        sent_size = recorded["input_tokens_sent"]
        assert sent_size < original_size, (
            f"input_tokens_sent ({sent_size}) should be smaller than the "
            f"original ({original_size}) once windowing has fired — if "
            f"they're equal, sent tokens are still being measured before "
            f"windowing runs"
        )


class TestCacheHitIsExplicitNotInferred:

    def test_cache_hit_true_on_actual_cache_hit(self, client, monkeypatch):
        c, fake = client
        monkeypatch.setattr(app_module.settings.cache, "enabled", True)

        from time import time as _time

        from tokenmizer.semantic_cache.cache import CacheEntry
        monkeypatch.setattr(
            app_module._cache, "get",
            lambda *a, **k: CacheEntry(key="k", prompt="p", response="cached!",
                                       input_tokens=1, output_tokens=1, created_at=_time()),
        )
        recorded = {}
        monkeypatch.setattr(app_module._analytics, "record",
                            lambda **kwargs: recorded.update(kwargs))

        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "cached question"}],
        })
        assert r.status_code == 200
        assert recorded["cache_hit"] is True

    def test_cache_hit_false_on_real_provider_call(self, client, monkeypatch):
        c, fake = client
        recorded = {}
        monkeypatch.setattr(app_module._analytics, "record",
                            lambda **kwargs: recorded.update(kwargs))
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "not cached"}],
        })
        assert r.status_code == 200
        assert recorded["cache_hit"] is False

    def test_cache_hit_false_even_if_provider_reports_zero_input_tokens(self, client, monkeypatch):
        """The old inference (input_tokens_actual == 0) would misfire
        here — a real provider call that happens to report 0 input
        tokens is NOT a cache hit."""
        c, fake = client
        fake.chat = AsyncMock(return_value=LLMResponse(
            text="answer", input_tokens=0, output_tokens=5,
            model="fake", provider="fake", latency_ms=1.0,
        ))
        recorded = {}
        monkeypatch.setattr(app_module._analytics, "record",
                            lambda **kwargs: recorded.update(kwargs))
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert r.status_code == 200
        assert recorded["cache_hit"] is False, (
            "a real provider response reporting 0 input tokens was "
            "misrecorded as a cache hit — cache_hit must be explicit, "
            "not inferred from token counts"
        )


class TestProviderErrorDoesNotLeakDetail:

    def test_502_does_not_echo_raw_exception_text(self, client):
        c, fake = client
        fake.chat = AsyncMock(side_effect=RuntimeError(
            "connection to https://internal-provider.example/v1/x?key=SECRET failed"
        ))
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert r.status_code == 502
        assert "SECRET" not in r.text
        assert "internal-provider.example" not in r.text
