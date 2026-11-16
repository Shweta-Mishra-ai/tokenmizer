"""Text a client may legally send must not fail the request.

A lone UTF-16 surrogate is valid JSON; it cannot be encoded as UTF-8, so a
request carrying one used to reach the provider adapter and fail with a 502
(found by benchmarks/load).
"""
from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from tokenmizer.api import app as app_module
from tokenmizer.providers.providers import LLMResponse


def test_a_lone_surrogate_is_not_a_provider_failure(monkeypatch):
    seen = []

    async def chat(messages, *a, **k):
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                c.encode("utf-8")              # what a real HTTP client must do
                seen.append(c)
        return LLMResponse(text="ok", input_tokens=1, output_tokens=1, model="m",
                           provider="fake", latency_ms=1.0)

    provider = AsyncMock()
    provider.chat = chat
    monkeypatch.setattr(app_module, "_get_provider", lambda: provider)
    monkeypatch.setattr(app_module.settings.cache, "enabled", False)
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
    raw = b'{"model": "m", "session_id": "sur", "messages": [{"role": "user", "content": "x \\ud800 y"}]}'
    with TestClient(app_module.app) as c:
        r = c.post("/v1/chat/completions", content=raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 200, r.text
    assert any("�" in s for s in seen)
