"""
`model_map`: send a client's model name to a different model.

This replaces the `routing` block, which was accepted by the config and
implemented by nothing — `savings["routing"]` was a hardcoded 0 and no
request was ever routed. What people actually reach for is a rename: a
client hard-codes "gpt-4", or an agent framework pins a model you do not
run, and you want one place to redirect it.

Pinned here:
- the substitution happens, and it is what the provider is called with
- the response reports the original name so a surprising answer is
  traceable to the config rather than to the model
- it is exact-match and applied once, so a map cannot loop
- an unmapped model is untouched and says nothing
- `savings` no longer carries the always-zero `routing` key
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tokenmizer.api import app as app_module
from tokenmizer.api.app import app
from tokenmizer.providers.providers import LLMResponse


class _Recorder:
    """Minimal provider: records the model it was called with."""

    def __init__(self):
        self.models: list[str] = []

    async def chat(self, messages, model, **kw):
        self.models.append(model)
        return LLMResponse(text="ok", input_tokens=5, output_tokens=2,
                           model=model, provider="fake")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
    monkeypatch.setattr(app_module.settings.cache, "enabled", False)
    with TestClient(app) as c:
        yield c


def _post(client, monkeypatch, provider, model, session="mm"):
    monkeypatch.setattr(app_module, "_get_provider", lambda: provider)
    return client.post("/v1/chat/completions", json={
        "model": model, "session_id": session,
        "messages": [{"role": "user", "content": "hello there, how are you"}],
    })


class TestModelMap:

    def test_mapped_model_is_what_the_provider_is_called_with(
            self, client, monkeypatch):
        monkeypatch.setattr(app_module.settings, "model_map",
                            {"gpt-4": "claude-sonnet-4-6"})
        p = _Recorder()
        r = _post(client, monkeypatch, p, "gpt-4", session="mm-1")

        assert r.status_code == 200, r.text
        assert p.models == ["claude-sonnet-4-6"], (
            "the map must change the model actually sent to the provider, "
            "not just the label in the response"
        )

    def test_response_says_what_it_was_mapped_from(self, client, monkeypatch):
        monkeypatch.setattr(app_module.settings, "model_map",
                            {"gpt-4": "claude-sonnet-4-6"})
        r = _post(client, monkeypatch, _Recorder(), "gpt-4", session="mm-2")
        body = r.json()

        assert body["model"] == "claude-sonnet-4-6"
        assert body["tokenmizer"]["model_mapped_from"] == "gpt-4", (
            "a client that asked for one model and got an answer from "
            "another has to be able to see why"
        )

    def test_unmapped_model_is_untouched_and_silent(self, client, monkeypatch):
        monkeypatch.setattr(app_module.settings, "model_map",
                            {"gpt-4": "claude-sonnet-4-6"})
        p = _Recorder()
        r = _post(client, monkeypatch, p, "claude-haiku-4-5", session="mm-3")

        assert p.models == ["claude-haiku-4-5"]
        assert "model_mapped_from" not in r.json()["tokenmizer"], (
            "the key is a signal that something was substituted; it must "
            "not appear on every response"
        )

    def test_map_is_applied_once_so_it_cannot_loop(self, client, monkeypatch):
        """a -> b and b -> a is a config mistake, not a hang."""
        monkeypatch.setattr(app_module.settings, "model_map",
                            {"a": "b", "b": "a"})
        p = _Recorder()
        r = _post(client, monkeypatch, p, "a", session="mm-4")

        assert r.status_code == 200
        assert p.models == ["b"]

    def test_empty_map_is_the_default(self):
        from tokenmizer.config.settings import Settings

        assert Settings().model_map == {}


class TestRoutingSavingsFieldIsGone:
    """`savings.routing` was always 0 — a layer reported on every response
    that had never done anything. Reporting a number for a layer that does
    not exist is worse than not reporting it."""

    def test_savings_has_no_routing_key(self, client, monkeypatch):
        r = _post(client, monkeypatch, _Recorder(), "claude-sonnet-4-6",
                  session="mm-5")

        assert "routing" not in r.json()["tokenmizer"]["savings"]
