"""
Regression tests for tokenmizer/providers/providers.py.

providers.py has 6 provider adapters and, before this file, zero
dedicated tests (20% coverage) — everything that exercised it did so
incidentally through analytics/audit tests that mock the provider away
entirely, never the adapter's own translation logic.

Bug covered here: AnthropicProvider._call()'s buffered-streaming branch
(reached via BaseProvider.chat(..., stream=True), a documented parameter
on the public class API even though nothing in the API layer currently
calls it that way — real streaming goes through chat_stream() instead)
computed input_tokens as `count_messages_tokens(conv, model)`, where
`conv` has the system message stripped out (Anthropic takes system as a
separate top-level param, not a message). Nothing added the system
prompt's tokens back in, so any caller using stream=True with a system
prompt got a silently undercounted input_tokens — corrupting cost/
savings analytics with no error, no log line, nothing. The non-streaming
branch never had this problem because it reads token counts straight off
`resp.usage`, the real API-reported figures. The fix makes the streaming
branch do the same via the SDK's `get_final_message()`.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from tokenmizer.providers.providers import AnthropicProvider, GeminiProvider


@dataclass
class _FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class _FakeFinalMessage:
    usage: _FakeUsage
    stop_reason: str = "end_turn"


class _FakeMessageStream:
    """Mimics anthropic's MessageStreamManager: async context manager
    exposing `.text_stream` and `.get_final_message()`."""

    def __init__(self, chunks: list[str], final: _FakeFinalMessage):
        self._chunks = chunks
        self._final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def _text_stream(self):
        for c in self._chunks:
            yield c

    @property
    def text_stream(self):
        return self._text_stream()

    async def get_final_message(self):
        return self._final


@pytest.mark.asyncio
async def test_anthropic_streaming_uses_real_usage_for_input_tokens(monkeypatch):
    import anthropic

    fake_final = _FakeFinalMessage(usage=_FakeUsage(input_tokens=999, output_tokens=3))

    class _FakeMessages:
        def stream(self, **kwargs):
            # A system prompt reaching the SDK call is exactly what the old
            # local estimate (count_messages_tokens(conv, ...)) dropped —
            # `conv` never includes it.
            assert "system" in kwargs
            return _FakeMessageStream(["Hel", "lo"], fake_final)

    class _FakeClient:
        def __init__(self, api_key=None):
            self.messages = _FakeMessages()

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _FakeClient)

    provider = AnthropicProvider(api_key="test-key")
    resp = await provider._call(
        messages=[{"role": "user", "content": "hi"}],
        model="claude-sonnet-4-6",
        max_tokens=100,
        stream=True,
        system="You are a helpful assistant. " * 200,
    )

    assert resp.text == "Hello"
    # 999 is nowhere near what a local estimate of the tiny "hi" message
    # alone would produce — this can only come from the real usage object,
    # proving the system prompt's tokens are no longer silently dropped.
    assert resp.input_tokens == 999
    assert resp.output_tokens == 3
    assert resp.finish_reason == "end_turn"


@dataclass
class _FakeGeminiUsage:
    prompt_token_count: int
    candidates_token_count: int


class _FakeGeminiResponse:
    def __init__(self, text: str, usage: _FakeGeminiUsage):
        self.text = text
        self.usage_metadata = usage


class _FakeChat:
    def __init__(self, response: _FakeGeminiResponse):
        self._response = response

    async def send_message_async(self, *args, **kwargs):
        return self._response


class _FakeGenerativeModel:
    def __init__(self, response: _FakeGeminiResponse, **kwargs):
        self._response = response
        # Constructor is called with system_instruction=... when a system
        # prompt is present — the bug this test guards against is that
        # system_instruction's tokens never made it into input_tokens.
        self.received_kwargs = kwargs

    def start_chat(self, history=None):
        return _FakeChat(self._response)


@pytest.mark.asyncio
async def test_gemini_uses_real_usage_metadata_for_input_tokens(monkeypatch):
    genai = pytest.importorskip("google.generativeai")

    fake_response = _FakeGeminiResponse(
        text="hi there",
        usage=_FakeGeminiUsage(prompt_token_count=1234, candidates_token_count=7),
    )
    fake_model = _FakeGenerativeModel(fake_response)

    monkeypatch.setattr(genai, "configure", lambda **kw: None)
    monkeypatch.setattr(genai, "GenerativeModel", lambda *a, **kw: fake_model)

    provider = GeminiProvider(api_key="test-key")
    resp = await provider._call(
        messages=[{"role": "user", "content": "hi"}],
        model="gemini-1.5-pro",
        max_tokens=100,
        stream=False,
        system="You are a helpful assistant. " * 200,
    )

    assert resp.text == "hi there"
    # 1234 is nowhere near a local estimate of "hi" alone — only the real
    # usage_metadata (which naturally includes system_instruction) gives
    # this figure, proving the fallback local estimate wasn't used.
    assert resp.input_tokens == 1234
    assert resp.output_tokens == 7


class TestRetryClassification:
    """Cohere, Gemini, and Ollama each had their own retry-classification
    bug — the same class OpenAI's _RETRYABLE_TEXT docstring already warns
    about, just never mirrored to the other providers:

      - Cohere: `"rate" in str(e).lower()` — a bare substring match.
        "rate" is inside "separate"/"moderate"/"generate", so an
        unrelated permanent error mentioning any of those words was
        retried 3 times for nothing, while a genuine rate-limit phrased
        as "too many requests" (no literal "rate") was never retried.
      - Gemini: only "quota"/"429" were recognized — every other
        retryable shape (timeout, overloaded, service unavailable) was
        wrongly treated as permanent for this provider only.
      - Ollama: retryable=True unconditionally, on every exception,
        including a malformed request or an unknown model name.
    """

    def test_cohere_uses_word_boundary_matching_not_bare_substring(self):
        """Cohere's fix reuses _RETRYABLE_TEXT directly (see the source),
        so this is really a test of that shared regex against the exact
        false-positive/false-negative pair from the bug."""
        from tokenmizer.providers.providers import _RETRYABLE_TEXT

        # False positive the old `"rate" in str(e).lower()` check made:
        assert not _RETRYABLE_TEXT.search("Invalid parameter: separate stop sequences required")
        # False negative the old check made:
        assert _RETRYABLE_TEXT.search("429 Too Many Requests")

    @pytest.mark.asyncio
    async def test_gemini_retries_on_timeout_not_just_quota(self, monkeypatch):
        genai = pytest.importorskip("google.generativeai")

        class _FakeChat:
            async def send_message_async(self, *a, **kw):
                raise RuntimeError("upstream request timed out")

        class _FakeModel:
            def start_chat(self, history=None):
                return _FakeChat()

        monkeypatch.setattr(genai, "configure", lambda **kw: None)
        monkeypatch.setattr(genai, "GenerativeModel", lambda *a, **kw: _FakeModel())

        provider = GeminiProvider(api_key="test-key")
        with pytest.raises(Exception) as exc_info:
            await provider._call(
                messages=[{"role": "user", "content": "hi"}],
                model="gemini-1.5-pro", max_tokens=10, stream=False, system="",
            )
        assert exc_info.value.retryable is True, (
            "a timeout with no 'quota'/'429' in the message must still retry"
        )


class TestOllamaRetryClassification:

    def test_client_side_timeout_is_retryable(self):
        import httpx

        from tokenmizer.providers.providers import _ollama_error_is_retryable
        assert _ollama_error_is_retryable(httpx.TimeoutException("timed out")) is True

    def test_connect_error_is_retryable(self):
        """Local server not up yet — the single most common real-world
        Ollama failure mode, and worth retrying unconditionally."""
        import httpx

        from tokenmizer.providers.providers import _ollama_error_is_retryable
        assert _ollama_error_is_retryable(httpx.ConnectError("connection refused")) is True

    def test_5xx_status_is_retryable(self):
        import httpx

        from tokenmizer.providers.providers import _ollama_error_is_retryable
        req = httpx.Request("POST", "http://localhost:11434/api/chat")
        resp = httpx.Response(503, request=req)
        err = httpx.HTTPStatusError("server error", request=req, response=resp)
        assert _ollama_error_is_retryable(err) is True

    def test_4xx_status_is_not_retryable(self):
        """A bad request (e.g. unknown model name) must not be retried —
        this is exactly the case the old unconditional retryable=True
        got wrong."""
        import httpx

        from tokenmizer.providers.providers import _ollama_error_is_retryable
        req = httpx.Request("POST", "http://localhost:11434/api/chat")
        resp = httpx.Response(400, request=req)
        err = httpx.HTTPStatusError("bad request", request=req, response=resp)
        assert _ollama_error_is_retryable(err) is False

    def test_generic_error_with_no_retryable_text_is_not_retryable(self):
        from tokenmizer.providers.providers import _ollama_error_is_retryable
        assert _ollama_error_is_retryable(ValueError("model 'nonexistent' not found")) is False
