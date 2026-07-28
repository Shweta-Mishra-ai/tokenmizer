"""
Regression tests for TM-05: the rate limiter's bucket key must not be a
raw client-supplied header value.

Bug: `client_id = request.headers.get("Authorization", client_ip)` used
whatever string the CLIENT sent as `Authorization` as the rate-limit
identity. A client can vary that header per request (trivially, and for
free in dev mode where no real key is even checked), landing in a fresh
bucket every time and never being limited at all — the opposite of what
a rate limiter is for.

Fix: key on `request.client.host` (the actual TCP-level source of the
request) always. It isn't something the caller can vary per-request the
way an arbitrary header string can be, unlike the old key.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from tokenmizer.api import app as app_module
from tokenmizer.api.app import app
from tokenmizer.providers.providers import LLMResponse


def _fake_response() -> LLMResponse:
    return LLMResponse(text="ok", input_tokens=1, output_tokens=1,
                       model="fake", provider="fake", latency_ms=1.0)


@pytest.fixture
def client(monkeypatch):
    fake_provider = AsyncMock()
    fake_provider.chat = AsyncMock(return_value=_fake_response())
    monkeypatch.setattr(app_module, "_get_provider", lambda: fake_provider)
    monkeypatch.setattr(app_module.settings.cache, "enabled", False)
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
    monkeypatch.setattr(app_module.settings.graph_checkpoint, "enabled", False)
    with TestClient(app) as c:
        yield c


class TestRateLimitKeyIsNotClientSupplied:

    def test_varying_authorization_header_does_not_reset_the_bucket(self, client, monkeypatch):
        seen_client_ids = []

        real_check = app_module._rate_limiter.check

        async def _spy_check(client_id):
            seen_client_ids.append(client_id)
            return await real_check(client_id)

        monkeypatch.setattr(app_module._rate_limiter, "check", _spy_check)

        for i in range(3):
            client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": f"Bearer totally-different-value-{i}"},
            )

        assert len(seen_client_ids) == 3
        assert len(set(seen_client_ids)) == 1, (
            f"varying the Authorization header produced different rate-limit "
            f"bucket keys ({set(seen_client_ids)!r}) — a client can bypass "
            f"rate limiting entirely by changing this header per request"
        )

    def test_key_is_derived_from_client_host_not_header_content(self, client, monkeypatch):
        captured = {}

        real_check = app_module._rate_limiter.check

        async def _spy_check(client_id):
            captured["client_id"] = client_id
            return await real_check(client_id)

        monkeypatch.setattr(app_module._rate_limiter, "check", _spy_check)

        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer some-arbitrary-client-string"},
        )
        assert "some-arbitrary-client-string" not in captured.get("client_id", ""), (
            "rate-limit key must never contain the raw Authorization header value"
        )


class TestApiRoutesAreRateLimited:
    """
    Previously only /v1/chat/completions called _check_rate_limit() at
    all — every /api/* endpoint (stats, graph, checkpoints, resume, ...)
    was completely unlimited. Since _check_rate_limit is already
    dependency-shaped (takes Request, raises HTTPException), it should be
    attached as a FastAPI dependency on those routes too.
    """

    def test_api_stats_endpoint_is_rate_limited(self, client, monkeypatch):
        from tokenmizer.api.rate_limiter import RateLimiter

        # A tiny limiter so the test doesn't need 60+ requests to trip it.
        tiny_limiter = RateLimiter(rate=2, per_seconds=60, burst=0)
        monkeypatch.setattr(app_module, "_rate_limiter", tiny_limiter)

        statuses = [client.get("/api/stats").status_code for _ in range(4)]
        assert 429 in statuses, (
            "/api/stats never returned 429 even after exceeding the rate "
            "limit — /api/* routes must be rate-limited, not just "
            "/v1/chat/completions"
        )
