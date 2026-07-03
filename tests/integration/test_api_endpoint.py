"""
End-to-end tests for the main proxy endpoint.

WHY THIS FILE EXISTS: v0.2.3 shipped with the @app.post decorator for
/v1/chat/completions accidentally attached to a helper function instead of
chat_completions() — the flagship endpoint returned null for every request,
and nothing caught it because no test ever POSTed to the app. These tests
exercise the real HTTP path so a route-registration regression can never
ship silently again.
"""
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from tokenmizer.api import app as app_module
from tokenmizer.api.app import app
from tokenmizer.providers.providers import LLMResponse


def _fake_response(text: str = "hello from fake provider") -> LLMResponse:
    return LLMResponse(
        text=text, input_tokens=10, output_tokens=5,
        model="fake-model", provider="fake", latency_ms=1.0,
    )


@pytest.fixture
def client(monkeypatch, tmp_path):
    """TestClient with a mocked provider and side-effect-free settings."""
    fake_provider = AsyncMock()
    fake_provider.chat = AsyncMock(return_value=_fake_response())

    # No real network, no cache interference, checkpoints to tmp dir
    monkeypatch.setattr(app_module, "_get_provider", lambda: fake_provider)
    monkeypatch.setattr(app_module.settings.cache, "enabled", False)
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)

    with TestClient(app) as c:
        yield c, fake_provider


class TestRouteRegistration:
    """The tests that would have caught the v0.2.3 decorator bug."""

    def test_chat_completions_route_is_registered(self):
        routes = {
            (r.path, m)
            for r in app.routes
            for m in getattr(r, "methods", set())
        }
        assert ("/v1/chat/completions", "POST") in routes

    def test_route_is_bound_to_chat_completions_not_a_helper(self):
        for r in app.routes:
            if getattr(r, "path", "") == "/v1/chat/completions":
                assert r.endpoint.__name__ == "chat_completions", (
                    f"POST /v1/chat/completions is bound to "
                    f"{r.endpoint.__name__!r} — decorator moved off the "
                    f"real handler again"
                )
                return
        pytest.fail("/v1/chat/completions route not found at all")


class TestChatCompletionsE2E:

    def test_basic_request_returns_openai_shape(self, client):
        c, _ = client
        r = c.post("/v1/chat/completions", json={
            "model": "claude-fable-5",
            "messages": [{"role": "user", "content": "Build a FastAPI auth service"}],
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "hello from fake provider"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert "usage" in body and "tokenmizer" in body

    def test_session_id_activates_pipeline_and_is_echoed(self, client):
        c, _ = client
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Use PostgreSQL for the database layer"}],
            "session_id": "e2e-test-session",
        })
        assert r.status_code == 200, r.text
        assert r.json()["session_id"] == "e2e-test-session"
        assert "checkpoint" in r.json()["tokenmizer"]

    def test_content_blocks_accepted_not_422(self, client):
        """OpenAI multimodal content-block format must not be rejected."""
        c, _ = client
        r = c.post("/v1/chat/completions", json={
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "summarize this"},
                    {"type": "text", "text": "second block"},
                ],
            }],
        })
        assert r.status_code == 200, r.text

    def test_unknown_openai_fields_accepted_not_422(self, client):
        """Standard OpenAI clients send fields we don't use — never 422."""
        c, _ = client
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi there friend"}],
            "frequency_penalty": 0.5,
            "presence_penalty": 0.1,
            "n": 1,
            "user": "abc",
        })
        assert r.status_code == 200, r.text

    def test_sampling_params_forwarded_to_provider(self, client):
        c, fake = client
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "deterministic please"}],
            "temperature": 0.0,
            "top_p": 0.9,
        })
        assert r.status_code == 200, r.text
        kwargs = fake.chat.call_args.kwargs
        assert kwargs.get("temperature") == 0.0
        assert kwargs.get("top_p") == 0.9

    def test_streaming_returns_sse_chunks(self, client):
        """v0.3: stream=true must return true SSE passthrough in OpenAI
        chat.completion.chunk format, ending with data: [DONE]."""
        c, fake = client

        async def fake_stream(**kwargs):
            for piece in ("Hello", " streamed", " world"):
                yield piece

        fake.chat_stream = fake_stream
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "stream me a story please"}],
            "stream": True,
        })
        assert r.status_code == 200, r.text
        assert "text/event-stream" in r.headers["content-type"]
        body = r.text
        assert "chat.completion.chunk" in body
        assert '"content": "Hello"' in body
        assert '"content": " streamed"' in body
        assert '"finish_reason": "stop"' in body
        assert body.rstrip().endswith("data: [DONE]")

    def test_streaming_unsupported_provider_returns_501(self, client, monkeypatch):
        """Providers without chat_stream override must get a clear 501,
        not a fake buffered stream."""
        from tokenmizer.api import app as app_module
        from tokenmizer.providers.providers import BaseProvider

        class NoStreamProvider(BaseProvider):
            async def _call(self, *a, **k):  # pragma: no cover
                raise NotImplementedError

        monkeypatch.setattr(app_module, "_get_provider", lambda: NoStreamProvider())
        c, _ = client
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert r.status_code == 501
        assert "stream" in r.json()["detail"].lower()

    def test_provider_error_returns_502(self, client):
        c, fake = client
        fake.chat = AsyncMock(side_effect=RuntimeError("provider exploded"))
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert r.status_code == 502

    def test_empty_messages_rejected(self, client):
        c, _ = client
        r = c.post("/v1/chat/completions", json={"messages": []})
        # Must not 500 — either validation error or handled response
        assert r.status_code != 500


class TestHealthAndDocs:

    def test_health(self, client):
        c, _ = client
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_graph_share_html(self, client):
        """Shareable graph page: real HTML with D3 + the session's nodes."""
        c, _ = client
        # Put something in the graph via the pipeline first
        c.post("/v1/chat/completions", json={
            "messages": [{"role": "user",
                          "content": "Decided: use PostgreSQL for the html-viz session"}],
            "session_id": "html-viz-test",
        })
        r = c.get("/api/graph/html-viz-test/html")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "d3" in r.text and "forceSimulation" in r.text
        assert "html-viz-test" in r.text
