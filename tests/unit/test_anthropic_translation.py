"""
Regressions found by driving the proxy end to end through the real
anthropic SDK against a server that applies the Messages API's request
rules, and through the CLI against a live proxy.
"""
from __future__ import annotations

import pytest

from tokenmizer.providers.tools import anthropic_messages

# ── Blank turns ──────────────────────────────────────────────────────────────
# Anthropic rejects a text block with no non-whitespace text; OpenAI clients
# routinely send "" (most often on assistant turns). The client got a 502 for
# a conversation that is valid in its own format.

def test_a_blank_earlier_turn_is_dropped():
    out = anthropic_messages([
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "b"},
    ])
    assert [m["content"] for m in out] == ["a", "b"]


def test_a_whitespace_first_turn_does_not_leave_an_assistant_first():
    out = anthropic_messages([
        {"role": "user", "content": "   "},
        {"role": "assistant", "content": "?"},
        {"role": "user", "content": "go"},
    ])
    assert out[0]["role"] == "user"
    assert out[-1]["content"] == "go"


def test_blank_text_blocks_in_a_list_count_as_blank_but_images_do_not():
    out = anthropic_messages([
        {"role": "user", "content": [{"type": "text", "text": " "}]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "text", "text": ""},
                                     {"type": "image_url", "image_url": {"url": "https://x/y.png"}}]},
    ])
    assert len(out) == 1 and out[0]["content"][-1]["type"] == "image"


def test_the_final_turn_is_kept_even_when_blank():
    out = anthropic_messages([{"role": "user", "content": "a"},
                              {"role": "assistant", "content": "b"},
                              {"role": "user", "content": " "}])
    assert out[-1]["content"] == " "


def test_blank_text_beside_a_tool_call_is_left_out():
    out = anthropic_messages([
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": [{"type": "text", "text": ""}],
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "w", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
    ])
    assert [b["type"] for b in out[1]["content"]] == ["tool_use"]


# ── Sampling parameters the installed SDK no longer has ─────────────────────

def test_sampling_params_the_sdk_lacks_are_dropped_not_fatal():
    """anthropic 2.x removed temperature and top_p: every request that set
    one failed in the SDK with "unexpected keyword argument" before anything
    was sent."""
    from tokenmizer.providers.providers import _anthropic_sampling

    def create(*, model, messages, max_tokens, stop_sequences=None, system=None):
        pass

    out = _anthropic_sampling({"temperature": 0.3, "top_p": 0.9, "stop": "END"}, create)
    assert out == {"stop_sequences": ["END"]}


def test_temperature_is_clamped_where_the_sdk_still_takes_it():
    from tokenmizer.providers.providers import _anthropic_sampling

    def create(*, model, messages, max_tokens, temperature=None, top_p=None):
        pass

    assert _anthropic_sampling({"temperature": 1.5, "top_p": 0.9}, create) == \
        {"temperature": 1.0, "top_p": 0.9}


def test_a_kwargs_method_is_assumed_to_take_everything():
    from tokenmizer.providers.providers import _anthropic_sampling
    assert _anthropic_sampling({"temperature": 0.2}, lambda **kw: None) == {"temperature": 0.2}


@pytest.mark.asyncio
async def test_a_request_with_temperature_reaches_the_installed_sdk(monkeypatch):
    """End to end through AnthropicProvider with a client whose create()
    has the installed SDK's real signature."""
    anthropic = pytest.importorskip("anthropic")
    import inspect

    from tokenmizer.providers.providers import AnthropicProvider

    real = inspect.signature(anthropic.resources.messages.AsyncMessages.create)
    sent = {}

    class Usage:
        input_tokens, output_tokens = 3, 1
        cache_read_input_tokens = cache_creation_input_tokens = 0

    class Resp:
        content = [type("B", (), {"type": "text", "text": "hi"})()]
        stop_reason = "end_turn"
        usage = Usage()

    class Messages:
        async def create(self, **kw):
            real.bind(None, **kw)          # raises TypeError on an unknown name
            sent.update(kw)
            return Resp()
    Messages.create.__signature__ = real

    class Client:
        messages = Messages()

    provider = AnthropicProvider(api_key="k")
    monkeypatch.setattr(provider, "_client", lambda *a, **k: Client())
    resp = await provider.chat([{"role": "user", "content": "hi"}],
                               model="claude-sonnet-4-6", temperature=0.4, top_p=0.5)
    assert resp.text == "hi"
    assert "model" in sent


# ── /api/stats?session_id= ───────────────────────────────────────────────────

def test_stats_for_a_session_are_that_sessions(monkeypatch):
    """With one API key the session_id was ignored: `tokenmizer stats
    <session>` printed the global totals for any id."""
    from fastapi.testclient import TestClient

    from tokenmizer.api import app as app_module
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
    monkeypatch.setattr(app_module.settings, "api_keys", [], raising=False)
    import uuid
    sid = f"stats-scope-{uuid.uuid4().hex[:8]}"   # analytics persist across runs
    for _ in range(2):
        app_module._analytics.record(session_id=sid, provider="fake", model="m",
                                     input_tokens_original=10, input_tokens_sent=10,
                                     output_tokens=1, tokens_saved=0, latency_ms=1.0,
                                     cache_hit=False)
    with TestClient(app_module.app) as c:
        mine = c.get("/api/stats", params={"session_id": sid}).json()
        none = c.get("/api/stats", params={"session_id": "never-used-session"}).json()
        everything = c.get("/api/stats").json()
    assert mine["total_requests"] == 2
    assert none["total_requests"] == 0
    assert everything["total_requests"] >= 2
