"""
Regressions from fuzzing the public API with malformed but valid JSON, and
from the MCP and dashboard paths that relay the proxy's answers. Each test
reproduces a failure seen before the fix.
"""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from tokenmizer.providers.providers import LLMResponse, ProviderError


class _RecordingProvider:
    """A provider that, like the real SDKs, must UTF-8 encode the request."""
    supports_tools = True
    supports_images = True
    supports_tool_stream = True

    def __init__(self, fail_stream: bool = False):
        self.sent: list = []
        self.fail_stream = fail_stream

    async def chat(self, messages, **kw):
        json.dumps(messages, ensure_ascii=False).encode("utf-8")
        self.sent.append(messages)
        return LLMResponse(text="ok", input_tokens=5, output_tokens=2,
                           model="m", provider="fake", latency_ms=1.0)

    async def chat_stream(self, messages, **kw):
        json.dumps(messages, ensure_ascii=False).encode("utf-8")
        yield "partial "
        if self.fail_stream:
            raise ProviderError("fake", "http_500", "upstream died", retryable=False)
        yield "answer"


@pytest.fixture
def proxy(monkeypatch):
    from tokenmizer.api import app as app_module
    provider = _RecordingProvider()
    monkeypatch.setattr(app_module, "_get_provider", lambda: provider)
    monkeypatch.setattr(app_module.settings.cache, "enabled", False)
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
    monkeypatch.setattr(app_module.settings, "api_keys", [], raising=False)
    with TestClient(app_module.app, raise_server_exceptions=False) as c:
        yield c, provider


def _post_raw(c, path, body: bytes, **kw):
    return c.post(path, content=body, headers={"content-type": "application/json"}, **kw)


# A lone surrogate is a valid JSON escape that UTF-8 cannot encode.
_LONE = b'"why does \\ud83d break?"'


def test_a_lone_surrogate_reaches_the_provider_as_text(proxy):
    """The SDK could not serialise the request, it was never sent, and the
    client got 502 "provider request failed" for its own input."""
    c, provider = proxy
    r = _post_raw(c, "/v1/chat/completions",
                  b'{"messages": [{"role": "user", "content": ' + _LONE + b'}]}')
    assert r.status_code == 200, r.text
    assert provider.sent[-1][-1]["content"] == "why does � break?"


def test_valid_emoji_are_left_alone(proxy):
    c, provider = proxy
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "ship it 😀"}]})
    assert r.status_code == 200
    assert provider.sent[-1][-1]["content"] == "ship it 😀"


def test_a_checkpoint_transcript_with_a_lone_surrogate_does_not_crash(proxy):
    """The resume text echoed it back and Starlette's UTF-8 render raised:
    a bare "Internal Server Error"."""
    c, _ = proxy
    r = _post_raw(c, "/api/checkpoint?session_id=sur-ckpt",
                  b'{"messages": [{"role": "user", "content": "Decided: use Redis ' + _LONE[1:-1] + b'"}]}')
    assert r.status_code == 200, r.text
    r.json()


def test_any_json_response_with_a_lone_surrogate_renders():
    from tokenmizer.api.app import _SafeJSONResponse
    body = _SafeJSONResponse({"q": "a\ud83db"}).body
    assert json.loads(body) == {"q": "a\ud83db"}
    assert _SafeJSONResponse({"q": "ok 😀"}).body == '{"q":"ok 😀"}'.encode()


def test_an_unreadable_pdf_is_read_as_text_not_a_500(proxy):
    """pypdf raised "Stream has ended unexpectedly" and /api/analyze
    answered 500 for the caller's own malformed file."""
    c, _ = proxy
    r = c.post("/api/analyze", json={"filename": "report.pdf",
                                     "content": "%PDF-1.4 truncated before any object"})
    assert r.status_code == 200, r.text
    assert r.json()["strategy_used"] == "text_fallback"
    assert "truncated" in r.json()["content"]


def test_a_failed_stream_does_not_claim_it_finished(proxy, monkeypatch):
    """After the error event the stream sent finish_reason "stop", which a
    client that ignores unknown keys read as a complete answer."""
    c, provider = proxy
    provider.fail_stream = True
    with c.stream("POST", "/v1/chat/completions",
                  json={"stream": True, "messages": [{"role": "user", "content": "hi"}]}) as s:
        events = [line[6:] for line in s.iter_lines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    parsed = [json.loads(e) for e in events[:-1]]
    assert any("error" in e for e in parsed)
    finishes = [e["choices"][0]["finish_reason"] for e in parsed if e.get("choices")]
    assert "stop" not in finishes


def _mcp_message(code: int, body: dict | None) -> str:
    from tokenmizer.mcp import server as mcp
    request = httpx.Request("GET", "http://localhost:8000/api/graph/s")
    response = (httpx.Response(code, json=body, request=request) if body is not None
                else httpx.Response(code, request=request))
    return mcp._explain(httpx.HTTPStatusError("e", request=request, response=response))


def test_mcp_relays_what_to_change_for_a_4xx():
    """ "TokenMizer returned HTTP 421." named no setting to change."""
    msg = _mcp_message(421, {"detail": "Host 'h' is not trusted. Set api_key, or list "
                                       "this host in security.trusted_hosts."})
    assert "security.trusted_hosts" in msg
    assert "Retry after" in _mcp_message(429, {"detail": "Rate limit exceeded. Retry after 1.0s"})


def test_mcp_keeps_5xx_detail_out_of_the_reply():
    msg = _mcp_message(500, {"detail": "Manual checkpoint failed for session acme-secret (ref: x)"})
    assert "acme-secret" not in msg


def test_mcp_401_always_says_where_the_key_goes():
    assert "TOKENMIZER_API_KEY" in _mcp_message(401, {"detail": "Missing API key"})


def test_the_key_mode_graph_tab_is_sandboxed():
    """With an API key the graph page was written straight into a tab of
    the dashboard's origin, which starts with a copy of its sessionStorage."""
    from tokenmizer.dashboard.page import DASHBOARD_HTML
    body = DASHBOARD_HTML.split("async function openGraph", 1)[1].split("\nlet ", 1)[0]
    assert "doc.write(html)" not in body
    assert "setAttribute('sandbox', 'allow-scripts allow-downloads')" in body
    assert "allow-same-origin" not in body
