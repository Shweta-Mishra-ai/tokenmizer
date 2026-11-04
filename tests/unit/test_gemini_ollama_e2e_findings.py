"""
Found by driving the google-genai SDK through the proxy against a server
applying Gemini's request rules, and the Ollama adapter against a fake
Ollama: parallel tool results rejected, images dropped, a client per
request, and a mid-stream Ollama failure passed off as a finished answer.
"""
from __future__ import annotations

import asyncio
import base64

import pytest

from tokenmizer.providers.tools import gemini_contents, gemini_parts

CALLS = [{"id": "a", "type": "function", "function": {"name": "weather", "arguments": '{"city":"Pune"}'}},
         {"id": "b", "type": "function", "function": {"name": "weather", "arguments": '{"city":"Goa"}'}}]


def test_parallel_tool_results_share_one_user_turn():
    """Gemini rejects a turn whose function_response count differs from the
    function_call count before it: one turn per result failed with 400."""
    out = gemini_contents([
        {"role": "user", "content": "both"},
        {"role": "assistant", "content": None, "tool_calls": CALLS},
        {"role": "tool", "tool_call_id": "a", "content": "rain"},
        {"role": "tool", "tool_call_id": "b", "content": "sun"},
        {"role": "user", "content": "thanks"},
    ])
    assert [c["role"] for c in out] == ["user", "model", "user", "user"]
    assert [list(p) for p in out[2]["parts"]] == [["function_response"]] * 2
    assert out[3]["parts"] == [{"text": "thanks"}]


def test_a_base64_image_becomes_inline_data():
    png = b"\x89PNG\r\n"
    parts = gemini_parts([
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
    ])
    assert parts == [{"text": "what is this"},
                     {"inline_data": {"mime_type": "image/png", "data": png}}]


def test_an_image_by_url_is_left_out_not_fetched(caplog):
    parts = gemini_parts([{"type": "text", "text": "x"},
                          {"type": "image_url", "image_url": {"url": "http://169.254.169.254/latest"}}])
    assert parts == [{"text": "x"}]
    assert "not sent to Gemini" in caplog.text


def test_gemini_accepts_images():
    from tokenmizer.providers.providers import GeminiProvider
    assert GeminiProvider.supports_images


def test_gemini_reuses_its_sdk_client(monkeypatch):
    genai = pytest.importorskip("google.genai")
    from tokenmizer.providers.providers import GeminiProvider
    built = []

    class FakeClient:
        def __init__(self, **kw):
            built.append(kw)

    monkeypatch.setattr(genai, "Client", FakeClient)
    p = GeminiProvider(api_key="k")

    async def twice():
        return p._genai_client(genai), p._genai_client(genai)
    a, b = asyncio.run(twice())
    assert a is b and len(built) == 1


def test_an_ollama_error_line_mid_stream_raises(monkeypatch):
    httpx = pytest.importorskip("httpx")
    from tokenmizer.providers.providers import OllamaProvider, ProviderError

    def handler(request):
        body = (b'{"message":{"content":"partial "},"done":false}\n'
                b'{"error":"model runner crashed"}\n')
        return httpx.Response(200, content=body)

    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw))
    p = OllamaProvider(base_url="http://ollama.test")

    async def drain():
        return [x async for x in p.chat_stream([{"role": "user", "content": "go"}], model="m")]
    with pytest.raises(ProviderError, match="model runner crashed"):
        asyncio.run(drain())
