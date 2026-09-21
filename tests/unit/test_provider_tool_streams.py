"""
Streamed tool calls on the four adapters that could not do them.

Before this, `stream: true` plus `tools` meant one of two things
depending on the provider:

  Anthropic, Ollama  the proxy fell back to a buffered non-streaming call
                     and re-chunked the finished answer. Valid SSE, but
                     the client waited for the whole thing.
  Gemini, Cohere     501. Both refused tools AND streaming outright,
                     though both SDKs have had them throughout.

All four now carry tool calls on the stream, mapped to the OpenAI delta
shape the proxy re-emits. Each provider numbers and fragments them
differently, and that mapping is what these tests pin:

  Anthropic  content_block_start(tool_use) then input_json_delta, indexed
             by CONTENT BLOCK — text and tool_use share one sequence.
  Ollama     finished calls on the last message, no fragments.
  Gemini     a whole function_call part per chunk, unnumbered.
  Cohere     tool-call-start then tool-call-delta, with its own index.

HONEST LIMITATION: these run against fakes shaped like each SDK's
documented objects, not against a live key. They pin the translation, and
they would not catch an SDK whose real objects differ from its docs.
"""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from tokenmizer.providers import tools as T
from tokenmizer.providers.providers import (
    AnthropicProvider,
    CohereProvider,
    GeminiProvider,
    OllamaProvider,
)


async def _collect(agen):
    return [chunk async for chunk in agen]


def _tool_deltas(chunks: list) -> list[dict]:
    return [d for c in chunks if isinstance(c, dict) for d in c["tool_calls"]]


def _text(chunks: list) -> str:
    return "".join(c for c in chunks if isinstance(c, str))


def _assembled(chunks: list) -> dict[int, dict]:
    """Assemble the deltas the way a client's SDK does: by index."""
    out: dict[int, dict] = {}
    for d in _tool_deltas(chunks):
        slot = out.setdefault(d["index"], {"id": "", "name": "", "arguments": ""})
        if d.get("id"):
            slot["id"] = d["id"]
        fn = d.get("function") or {}
        if fn.get("name"):
            slot["name"] = fn["name"]
        slot["arguments"] += fn.get("arguments") or ""
    return out


# ── Anthropic ────────────────────────────────────────────────────────────────

class _AnthropicStream:
    """The raw event stream, in the SDK's documented order. The tool_use
    block is at content-block index 1, behind a text block — which is the
    case that broke when the raw index was forwarded as the tool index."""

    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def __aiter__(self):
        async def gen():
            for e in self._events:
                yield e
        return gen()


ANTHROPIC_EVENTS = [
    NS(type="content_block_start", index=0, content_block=NS(type="text", text="")),
    NS(type="content_block_delta", index=0, delta=NS(type="text_delta", text="Let me check. ")),
    NS(type="content_block_stop", index=0),
    NS(type="content_block_start", index=1,
       content_block=NS(type="tool_use", id="toolu_abc", name="get_weather")),
    NS(type="content_block_delta", index=1,
       delta=NS(type="input_json_delta", partial_json='{"city":')),
    NS(type="content_block_delta", index=1,
       delta=NS(type="input_json_delta", partial_json='"Pune"}')),
    NS(type="content_block_stop", index=1),
]


@pytest.mark.asyncio
async def test_anthropic_streams_tool_calls(monkeypatch):
    anthropic = pytest.importorskip("anthropic")

    class _Messages:
        def stream(self, **kw):
            _Messages.seen = kw
            return _AnthropicStream(ANTHROPIC_EVENTS)

    monkeypatch.setattr(anthropic, "AsyncAnthropic",
                        lambda **kw: NS(messages=_Messages()))

    chunks = await _collect(AnthropicProvider(api_key="k").chat_stream(
        [{"role": "user", "content": "weather in Pune?"}],
        model="claude-sonnet-4-6",
        tools=[{"type": "function",
                "function": {"name": "get_weather", "parameters": {}}}],
    ))

    assert _text(chunks) == "Let me check. "
    assembled = _assembled(chunks)
    assert assembled == {0: {"id": "toolu_abc", "name": "get_weather",
                             "arguments": '{"city":"Pune"}'}}, (
        "the tool_use block is content-block 1; forwarding that raw index "
        "announces the call at index 1 with nothing at index 0, and every "
        "client SDK that assembles by index breaks"
    )


@pytest.mark.asyncio
async def test_anthropic_forwards_the_tool_declarations(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    seen: dict = {}

    class _Messages:
        def stream(self, **kw):
            seen.update(kw)
            return _AnthropicStream([])

    monkeypatch.setattr(anthropic, "AsyncAnthropic",
                        lambda **kw: NS(messages=_Messages()))

    await _collect(AnthropicProvider(api_key="k").chat_stream(
        [{"role": "user", "content": "hi"}], model="claude-sonnet-4-6",
        tools=[{"type": "function",
                "function": {"name": "get_weather", "description": "w",
                             "parameters": {"type": "object"}}}],
    ))

    assert seen["tools"] == [{"name": "get_weather", "description": "w",
                              "input_schema": {"type": "object"}}]


# ── Ollama ───────────────────────────────────────────────────────────────────

class _OllamaResponse:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


@pytest.mark.asyncio
async def test_ollama_streams_tool_calls_and_sends_the_declarations(monkeypatch):
    import httpx

    captured: dict = {}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, method, url, json=None):
            captured.update(json)
            return _OllamaResponse([
                json_line({"message": {"content": "Checking. "}}),
                json_line({"message": {"content": "", "tool_calls": [
                    {"function": {"name": "get_weather",
                                  "arguments": {"city": "Pune"}}}]},
                           "done": True}),
            ])

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    chunks = await _collect(OllamaProvider("llama3").chat_stream(
        [{"role": "user", "content": "weather?"}],
        tools=[{"type": "function",
                "function": {"name": "get_weather", "parameters": {}}}],
    ))

    assert _text(chunks) == "Checking. "
    assembled = _assembled(chunks)
    assert assembled[0]["name"] == "get_weather"
    assert json.loads(assembled[0]["arguments"]) == {"city": "Pune"}
    assert captured["tools"], (
        "the streaming payload carried no tools at all, so a client that "
        "asked for tools AND a stream got a model that could not see them"
    )


def json_line(obj) -> str:
    return json.dumps(obj)


# ── Gemini ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_gemini_streams_text_and_tool_calls(monkeypatch):
    genai = pytest.importorskip("google.genai")
    seen: dict = {}

    chunks_in = [
        NS(candidates=[NS(content=NS(parts=[NS(text="Let me check. ",
                                               function_call=None)]))]),
        NS(candidates=[NS(content=NS(parts=[
            NS(text=None, function_call=NS(name="get_weather",
                                           args={"city": "Pune"}))]))]),
    ]

    class _Models:
        async def generate_content_stream(self, **kw):
            seen.update(kw)

            async def gen():
                for c in chunks_in:
                    yield c
            return gen()

    monkeypatch.setattr(genai, "Client", lambda **kw: NS(aio=NS(models=_Models())))

    chunks = await _collect(GeminiProvider(api_key="k").chat_stream(
        [{"role": "user", "content": "weather in Pune?"}],
        model="gemini-2.5-flash",
        tools=[{"type": "function",
                "function": {"name": "get_weather", "parameters": {}}}],
    ))

    assert _text(chunks) == "Let me check. "
    assembled = _assembled(chunks)
    assert assembled[0]["name"] == "get_weather"
    assert json.loads(assembled[0]["arguments"]) == {"city": "Pune"}
    assert assembled[0]["id"], "Gemini mints no id and the client needs one"
    assert seen["config"].tools, "the declarations never reached the SDK"


# ── Cohere ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cohere_streams_text_and_tool_calls(monkeypatch):
    cohere = pytest.importorskip("cohere")
    seen: dict = {}

    events = [
        NS(type="content-start", index=0, delta=None),
        NS(type="content-delta", index=0,
           delta=NS(message=NS(content=NS(text="Let me check. ")))),
        NS(type="tool-call-start", index=0,
           delta=NS(message=NS(tool_calls=NS(
               id="call_c1", function=NS(name="get_weather", arguments=""))))),
        NS(type="tool-call-delta", index=0,
           delta=NS(message=NS(tool_calls=NS(
               id=None, function=NS(name=None, arguments='{"city":"Pune"}'))))),
        NS(type="message-end", index=None, delta=None),
    ]

    class _Client:
        def chat_stream(self, **kw):
            seen.update(kw)

            async def gen():
                for e in events:
                    yield e
            return gen()

    monkeypatch.setattr(cohere, "AsyncClientV2", lambda **kw: _Client())

    chunks = await _collect(CohereProvider(api_key="k").chat_stream(
        [{"role": "user", "content": "weather in Pune?"}],
        model="command-r-plus",
        tools=[{"type": "function",
                "function": {"name": "get_weather", "parameters": {}}}],
    ))

    assert _text(chunks) == "Let me check. "
    assembled = _assembled(chunks)
    assert assembled[0] == {"id": "call_c1", "name": "get_weather",
                            "arguments": '{"city":"Pune"}'}
    assert seen["tools"], "the declarations never reached the SDK"


# ── translation, without any SDK ──────────────────────────────────────────────

class TestGeminiTranslation:

    def test_tools(self):
        out = T.gemini_tools([{"type": "function", "function": {
            "name": "get_weather", "description": "w",
            "parameters": {"type": "object", "properties": {}}}}])
        assert out == [{"function_declarations": [{
            "name": "get_weather", "description": "w",
            "parameters": {"type": "object", "properties": {}}}]}]

    @pytest.mark.parametrize("choice,mode", [
        (None, None), ("auto", None), ("none", "NONE"), ("required", "ANY"),
    ])
    def test_tool_config(self, choice, mode):
        out = T.gemini_tool_config(choice)
        if mode is None:
            assert out is None, "auto is Gemini's own default"
        else:
            assert out["function_calling_config"]["mode"] == mode

    def test_named_function_narrows_to_that_name(self):
        out = T.gemini_tool_config(
            {"type": "function", "function": {"name": "get_weather"}})
        assert out["function_calling_config"]["allowed_function_names"] == ["get_weather"]

    def test_a_tool_round_trip_survives_the_conversation(self):
        """The second turn of an agent loop: without this, Gemini was sent
        a conversation with the call and its result missing, and asked for
        the same tool again."""
        out = T.gemini_contents([
            {"role": "user", "content": "weather in Pune?"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather",
                             "arguments": '{"city": "Pune"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 31}'},
        ])

        assert out[0] == {"role": "user", "parts": [{"text": "weather in Pune?"}]}
        assert out[1]["role"] == "model"
        assert out[1]["parts"][0]["function_call"] == {
            "name": "get_weather", "args": {"city": "Pune"}}
        # Gemini matches a response to a call by NAME, and the OpenAI shape
        # only carries the id — so the name is looked up from the request.
        assert out[2]["parts"][0]["function_response"]["name"] == "get_weather"


class TestCohereTranslation:

    def test_tool_result_becomes_a_document_part(self):
        out = T.cohere_messages([
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 31}'}])
        assert out[0]["tool_call_id"] == "call_1"
        assert out[0]["content"][0]["document"]["data"] == '{"temp_c": 31}'

    def test_a_tool_call_turn_never_carries_null_content(self):
        out = T.cohere_messages([
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "f", "arguments": "{}"}}]}])
        assert out[0]["content"] == ""

    def test_events_with_no_payload_are_ignored(self):
        for kind in ("content-start", "content-end", "message-start", "message-end"):
            assert T.cohere_stream_event(NS(type=kind, index=0, delta=None)) is None
