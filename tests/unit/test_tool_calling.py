"""
Tool/function calling through the proxy.

`tools`/`tool_choice` used to be accepted and dropped: the model answered
as if it had no tools and every agent framework that relies on function
calling broke silently behind the proxy. They are now forwarded — natively
for the OpenAI-compatible family, translated for Anthropic and Ollama
(providers/tools.py) — and `tool_calls` come back in the OpenAI shape, in
both the plain and the streamed response.

Rules pinned here:
- tool traffic survives every pipeline layer (redaction, compression,
  windowing, context injection) unchanged
- a tool-call answer is never cached and never output-trimmed
- a request that ends on a tool result is never served from the cache
- a provider without tool support refuses the request (501) instead of
  sending the model a conversation it cannot see the tools for
"""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from tokenmizer.api import app as app_module
from tokenmizer.api.app import app
from tokenmizer.providers import tools as T
from tokenmizer.providers.providers import (
    AnthropicProvider,
    LLMResponse,
    OllamaProvider,
    OpenAIProvider,
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}]

CALL = {"id": "call_abc", "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Pune"}'}}

# An agent loop's second request: the model asked for a tool, the client
# ran it, and now sends the result back. Ends on a tool turn, not a user one.
ROUND_TRIP = [
    {"role": "user", "content": "what's the weather in Pune?"},
    {"role": "assistant", "content": None, "tool_calls": [CALL]},
    {"role": "tool", "tool_call_id": "call_abc", "content": '{"temp_c": 31}'},
]


# ── translation ──────────────────────────────────────────────────────────────

class TestAnthropicTranslation:

    def test_tools(self):
        out = T.anthropic_tools(TOOLS)
        assert out == [{"name": "get_weather", "description": "Weather for a city",
                        "input_schema": TOOLS[0]["function"]["parameters"]}]

    @pytest.mark.parametrize("choice,expected", [
        (None, None),
        ("auto", None),
        ("required", {"type": "any"}),
        ("none", {"type": "none"}),
        ({"type": "function", "function": {"name": "get_weather"}},
         {"type": "tool", "name": "get_weather"}),
    ])
    def test_tool_choice(self, choice, expected):
        assert T.anthropic_tool_choice(choice) == expected

    def test_parallel_false_disables_parallel_use(self):
        assert T.anthropic_tool_choice("auto", parallel_tool_calls=False) == {
            "type": "auto", "disable_parallel_tool_use": True}

    def test_messages_round_trip(self):
        out = T.anthropic_messages(ROUND_TRIP + [
            {"role": "tool", "tool_call_id": "call_2", "content": "second"}])
        assert out[0] == {"role": "user", "content": "what's the weather in Pune?"}
        assert out[1] == {"role": "assistant", "content": [
            {"type": "tool_use", "id": "call_abc", "name": "get_weather",
             "input": {"city": "Pune"}}]}
        # Both results share ONE user message, as Anthropic requires.
        assert out[2]["role"] == "user"
        assert [b["type"] for b in out[2]["content"]] == ["tool_result", "tool_result"]
        assert out[2]["content"][0]["tool_use_id"] == "call_abc"
        assert "_tool_results" not in out[2]

    def test_assistant_text_and_call_both_kept(self):
        out = T.anthropic_messages([{"role": "assistant", "content": "checking",
                                     "tool_calls": [CALL]}])
        assert out[0]["content"][0] == {"type": "text", "text": "checking"}
        assert out[0]["content"][1]["type"] == "tool_use"

    def test_malformed_arguments_are_kept_not_dropped(self):
        out = T.anthropic_messages([{"role": "assistant", "content": "",
                                     "tool_calls": [{"id": "c", "type": "function",
                                                     "function": {"name": "f",
                                                                  "arguments": "{oops"}}]}])
        assert out[0]["content"][0]["input"] == {"_raw": "{oops"}

    def test_response_blocks(self):
        text, calls = T.anthropic_response([
            NS(type="text", text="Let me check."),
            NS(type="tool_use", id="toolu_1", name="get_weather", input={"city": "Pune"}),
        ])
        assert text == "Let me check."
        assert calls == [{"id": "toolu_1", "type": "function",
                          "function": {"name": "get_weather", "arguments": '{"city": "Pune"}'}}]


class TestOllamaTranslation:

    def test_arguments_become_a_dict_on_the_way_in(self):
        out = T.ollama_messages(ROUND_TRIP)
        assert out[1]["tool_calls"] == [{"function": {"name": "get_weather",
                                                      "arguments": {"city": "Pune"}}}]
        assert out[1]["content"] == ""
        assert out[2] == ROUND_TRIP[2]

    def test_arguments_become_a_string_on_the_way_out(self):
        calls = T.ollama_tool_calls([{"function": {"name": "get_weather",
                                                   "arguments": {"city": "Pune"}}}])
        assert calls[0]["function"] == {"name": "get_weather", "arguments": '{"city": "Pune"}'}
        assert calls[0]["id"].startswith("call_")


class TestShared:

    def test_normalize_accepts_sdk_objects(self):
        tc = NS(id="x", type="function", function=NS(name="f", arguments='{"a": 1}'))
        assert T.normalize_tool_calls([tc]) == [
            {"id": "x", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}]

    def test_finish_reason(self):
        assert T.finish_reason("stop", [CALL]) == "tool_calls"
        assert T.finish_reason("end_turn", []) == "stop"
        assert T.finish_reason("max_tokens", []) == "length"

    def test_strip_tool_fields_renders_calls_as_text(self):
        out = T.strip_tool_fields(ROUND_TRIP)
        assert out[1]["content"] == '[called tools: get_weather({"city": "Pune"})]'
        assert "tool_calls" not in out[1]
        assert out[2] == {"role": "tool", "content": '{"temp_c": 31}'}


# ── adapters ─────────────────────────────────────────────────────────────────

async def test_openai_forwards_tools_and_returns_tool_calls(monkeypatch):
    import openai

    seen = {}

    class _Completions:
        async def create(self, **kw):
            seen.update(kw)
            msg = NS(content=None, tool_calls=[
                NS(id="call_1", type="function",
                   function=NS(name="get_weather", arguments='{"city":"Pune"}'))])
            return NS(choices=[NS(message=msg, finish_reason="tool_calls")],
                      usage=NS(prompt_tokens=10, completion_tokens=5))

    class _Client:
        def __init__(self, **kw):
            self.chat = NS(completions=_Completions())

    monkeypatch.setattr(openai, "AsyncOpenAI", _Client)
    resp = await OpenAIProvider("k")._call(
        ROUND_TRIP, "gpt-4o", 100, False, "", tools=TOOLS, tool_choice="auto")
    assert seen["tools"] == TOOLS and seen["tool_choice"] == "auto"
    assert seen["messages"][1]["tool_calls"] == [CALL]     # passed through as-is
    assert seen["messages"][2]["tool_call_id"] == "call_abc"
    assert resp.tool_calls[0]["function"]["name"] == "get_weather"
    assert resp.finish_reason == "tool_calls"


async def test_openai_stream_yields_tool_call_deltas_as_events(monkeypatch):
    import openai

    chunks = [
        NS(choices=[NS(delta=NS(content="", tool_calls=[
            NS(index=0, id="call_1", type="function",
               function=NS(name="get_weather", arguments=""))]), finish_reason=None)]),
        NS(choices=[NS(delta=NS(content=None, tool_calls=[
            NS(index=0, id=None, type=None,
               function=NS(name=None, arguments='{"city":'))]), finish_reason=None)]),
        NS(choices=[NS(delta=NS(content="done", tool_calls=None), finish_reason="tool_calls")]),
    ]

    class _Completions:
        async def create(self, **kw):
            assert kw["tools"] == TOOLS

            async def gen():
                for c in chunks:
                    yield c
            return gen()

    class _Client:
        def __init__(self, **kw):
            self.chat = NS(completions=_Completions())

    monkeypatch.setattr(openai, "AsyncOpenAI", _Client)
    events = [e async for e in OpenAIProvider("k").chat_stream(
        [{"role": "user", "content": "hi"}], model="gpt-4o", tools=TOOLS)]
    assert events == [
        {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                         "function": {"name": "get_weather", "arguments": ""}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]},
        "done",
    ]


async def test_anthropic_sends_translated_tools_and_parses_tool_use(monkeypatch):
    import anthropic

    seen = {}

    class _Messages:
        async def create(self, **kw):
            seen.update(kw)
            return NS(content=[NS(type="text", text="Checking."),
                               NS(type="tool_use", id="toolu_1", name="get_weather",
                                  input={"city": "Pune"})],
                      usage=NS(input_tokens=10, output_tokens=5), stop_reason="tool_use")

    class _Client:
        def __init__(self, api_key=None):
            self.messages = _Messages()

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _Client)
    resp = await AnthropicProvider("k")._call(
        ROUND_TRIP, "claude-sonnet-4-6", 100, False, "",
        tools=TOOLS, tool_choice="required")
    assert seen["tools"][0]["input_schema"] == TOOLS[0]["function"]["parameters"]
    assert seen["tool_choice"] == {"type": "any"}
    assert seen["messages"][1]["content"][0]["type"] == "tool_use"
    assert seen["messages"][2]["content"][0]["type"] == "tool_result"
    assert resp.text == "Checking."
    assert resp.tool_calls[0]["function"] == {"name": "get_weather",
                                              "arguments": '{"city": "Pune"}'}
    assert resp.finish_reason == "tool_calls"


async def test_ollama_sends_tools_and_parses_tool_calls(monkeypatch):
    import httpx

    seen = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Pune"}}}]},
                "done_reason": "stop", "prompt_eval_count": 3, "eval_count": 2}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            seen.update(json)
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    resp = await OllamaProvider("llama3")._call(ROUND_TRIP, "llama3", 100, False, "",
                                               tools=TOOLS)
    assert seen["tools"] == TOOLS
    assert seen["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"city": "Pune"}
    assert resp.tool_calls[0]["function"]["arguments"] == '{"city": "Pune"}'
    assert resp.finish_reason == "tool_calls"


# ── proxy, non-streaming ─────────────────────────────────────────────────────

class _ToolProvider:
    supports_tools = True
    supports_tool_stream = False

    def __init__(self, answers):
        self.answers = list(answers)
        self.seen: list[dict] = []

    async def chat(self, messages, **kw):
        self.seen.append({"messages": messages, **kw})
        return self.answers.pop(0)


TOOL_ANSWER = LLMResponse(text="", input_tokens=10, output_tokens=5, model="m",
                          provider="fake", finish_reason="tool_calls", tool_calls=[CALL])
TEXT_ANSWER = LLMResponse(text="It is 31C in Pune.", input_tokens=10, output_tokens=5,
                          model="m", provider="fake")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
    monkeypatch.setattr(app_module.settings.cache, "enabled", True)
    app_module._cache.clear()
    with TestClient(app) as c:
        yield c


def _post(c, provider, monkeypatch, body):
    monkeypatch.setattr(app_module, "_get_provider", lambda: provider)
    r = c.post("/v1/chat/completions", json={"model": "gpt-4o", **body})
    return r


def test_tool_call_answer_is_returned_in_openai_shape(client, monkeypatch):
    provider = _ToolProvider([TOOL_ANSWER])
    r = _post(client, provider, monkeypatch, {
        "session_id": "tools-1", "tools": TOOLS,
        "messages": [{"role": "user", "content": "what's the weather in Pune?"}]})
    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"] == [CALL]
    assert choice["message"]["content"] is None
    assert provider.seen[0]["tools"] == TOOLS
    assert r.json()["tokenmizer"]["cache_hit"] is False
    assert "fallback" not in r.json()["tokenmizer"]


def test_tool_answer_is_never_cached(client, monkeypatch):
    provider = _ToolProvider([TOOL_ANSWER, TOOL_ANSWER])
    body = {"session_id": "tools-nocache", "tools": TOOLS,
            "messages": [{"role": "user", "content": "weather in Pune?"}]}
    _post(client, provider, monkeypatch, body)
    r = _post(client, provider, monkeypatch, body)
    assert r.json()["tokenmizer"]["cache_hit"] is False
    assert len(provider.seen) == 2, "the second identical tool request was served from cache"


def test_tool_result_round_trip_reaches_the_provider_intact(client, monkeypatch):
    provider = _ToolProvider([TEXT_ANSWER])
    r = _post(client, provider, monkeypatch, {
        "session_id": "tools-rt", "tools": TOOLS, "messages": ROUND_TRIP})
    assert r.status_code == 200, r.text
    sent = provider.seen[0]["messages"]
    conv = [m for m in sent if m.get("role") != "system"]
    assert conv[1]["tool_calls"] == [CALL]
    assert conv[2]["role"] == "tool" and conv[2]["tool_call_id"] == "call_abc"
    assert conv[2]["content"] == '{"temp_c": 31}'
    assert r.json()["choices"][0]["message"]["content"] == "It is 31C in Pune."
    assert r.json()["choices"][0]["finish_reason"] == "stop"


def test_request_ending_on_a_tool_result_is_not_served_from_cache(client, monkeypatch):
    # The final user turn is the same as an earlier cached one, but this
    # request ends on a tool result: a different question.
    first = _ToolProvider([TEXT_ANSWER])
    _post(client, first, monkeypatch, {
        "session_id": "tools-tail",
        "messages": [{"role": "user", "content": "what's the weather in Pune?"}]})
    second = _ToolProvider([TEXT_ANSWER])
    r = _post(client, second, monkeypatch, {"session_id": "tools-tail", "messages": ROUND_TRIP})
    assert r.json()["tokenmizer"]["cache_hit"] is False
    assert len(second.seen) == 1


def test_tool_messages_survive_compression_and_windowing(client, monkeypatch):
    monkeypatch.setattr(app_module.settings.memory, "max_tokens_before_summary", 400)
    monkeypatch.setattr(app_module._smart_window, "token_budget", 400)
    monkeypatch.setattr(app_module._smart_window, "protect_recent", 4)
    history = []
    for i in range(12):
        history.append({"role": "user", "content": f"turn {i} " + "filler " * 60})
        history.append({"role": "assistant", "content": f"reply {i} " + "filler " * 60})
    big_result = json.dumps({"rows": [{"id": i, "name": f"row {i}"} for i in range(60)]})
    messages = history + [
        {"role": "user", "content": "list the rows"},
        {"role": "assistant", "content": None, "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "call_abc", "content": big_result},
    ]
    provider = _ToolProvider([TEXT_ANSWER])
    r = _post(client, provider, monkeypatch, {
        "session_id": "tools-window", "tools": TOOLS, "messages": messages})
    assert r.status_code == 200, r.text
    sent = provider.seen[0]["messages"]
    tool_msgs = [m for m in sent if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["content"] == big_result, "tool result was altered"
    calls = [m for m in sent if m.get("tool_calls")]
    assert calls and calls[0]["tool_calls"] == [CALL]
    # Windowing engaged — the early turns are gone — and the tail is intact.
    assert len(sent) < len(messages)
    assert r.json()["tokenmizer"]["savings"].get("windowing", 0) > 0


def test_provider_without_tool_support_is_a_501(client, monkeypatch):
    class _NoTools:
        supports_tools = False

        async def chat(self, *a, **k):  # pragma: no cover — must not be reached
            raise AssertionError("request must be refused before the provider is called")

    r = _post(client, _NoTools(), monkeypatch, {
        "session_id": "tools-501", "tools": TOOLS,
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 501
    assert "Tool calling is not implemented" in r.json()["detail"]


def test_plain_request_shape_is_unchanged(client, monkeypatch):
    provider = _ToolProvider([TEXT_ANSWER])
    r = _post(client, provider, monkeypatch, {
        "session_id": "tools-plain",
        "messages": [{"role": "user", "content": "hello there"}]})
    choice = r.json()["choices"][0]
    assert "tool_calls" not in choice["message"]
    assert choice["finish_reason"] == "stop"
    assert "tools" not in provider.seen[0]
    for m in provider.seen[0]["messages"]:
        assert set(m) == {"role", "content"}


# ── proxy, streaming ─────────────────────────────────────────────────────────

def _events(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


def test_stream_passes_tool_call_deltas_through(client, monkeypatch):
    class _Streaming:
        supports_tools = True
        supports_tool_stream = True

        async def chat_stream(self, messages, **kw):
            assert kw["tools"] == TOOLS
            yield "Let me check. "
            yield {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                   "function": {"name": "get_weather", "arguments": ""}}]}
            yield {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":"Pune"}'}}]}

    monkeypatch.setattr(app_module, "_get_provider", lambda: _Streaming())
    r = client.post("/v1/chat/completions", json={
        "model": "gpt-4o", "session_id": "tools-stream", "stream": True, "tools": TOOLS,
        "messages": [{"role": "user", "content": "weather in Pune?"}]})
    assert r.status_code == 200, r.text
    events = _events(r.text)
    deltas = [e["choices"][0]["delta"] for e in events]
    assert {"content": "Let me check. "} in deltas
    tool_deltas = [d["tool_calls"] for d in deltas if "tool_calls" in d]
    assert tool_deltas[0][0]["id"] == "call_1"
    assert tool_deltas[1][0]["function"]["arguments"] == '{"city":"Pune"}'
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert r.text.strip().endswith("data: [DONE]")


def test_stream_is_buffered_for_a_provider_without_tool_streaming(client, monkeypatch):
    provider = _ToolProvider([LLMResponse(
        text="Checking.", input_tokens=1, output_tokens=1, model="m", provider="fake",
        finish_reason="tool_calls", tool_calls=[CALL])])
    monkeypatch.setattr(app_module, "_get_provider", lambda: provider)
    r = client.post("/v1/chat/completions", json={
        "model": "claude-sonnet-4-6", "session_id": "tools-buffered", "stream": True,
        "tools": TOOLS, "messages": [{"role": "user", "content": "weather in Pune?"}]})
    assert r.status_code == 200, r.text
    events = _events(r.text)
    deltas = [e["choices"][0]["delta"] for e in events]
    assert {"content": "Checking."} in deltas
    tool_delta = next(d["tool_calls"] for d in deltas if "tool_calls" in d)
    assert tool_delta[0]["function"] == CALL["function"] and tool_delta[0]["index"] == 0
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert provider.seen[0]["tools"] == TOOLS


# ── layers ───────────────────────────────────────────────────────────────────

def test_compression_leaves_tool_turns_untouched():
    from tokenmizer.compression.engine import CompressionPipeline

    noisy = "Certainly!   Here   is   the   result.  " * 40
    messages = [
        {"role": "user", "content": noisy},
        {"role": "assistant", "content": noisy, "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "call_abc", "content": noisy},
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "y"},
        {"role": "user", "content": "z"},
    ]
    out, _ = CompressionPipeline(min_tokens_to_compress=10).compress_messages(
        messages, protect_recent=3)
    assert out[1] == messages[1]
    assert out[2] == messages[2]


def test_tool_calls_count_toward_context_size():
    from tokenmizer.core.tokenizer import count_messages_tokens

    without = count_messages_tokens([{"role": "assistant", "content": ""}])
    with_call = count_messages_tokens([{"role": "assistant", "content": "",
                                        "tool_calls": [CALL]}])
    assert with_call > without


def test_chat_message_to_dict_only_adds_tool_fields_when_set():
    plain = app_module.ChatMessage(role="user", content="hi").to_dict()
    assert plain == {"role": "user", "content": "hi"}
    tool = app_module.ChatMessage(role="tool", content="r", tool_call_id="c1").to_dict()
    assert tool == {"role": "tool", "content": "r", "tool_call_id": "c1"}
    call = app_module.ChatMessage(role="assistant", content=None, tool_calls=[CALL]).to_dict()
    assert call == {"role": "assistant", "content": "", "tool_calls": [CALL]}
