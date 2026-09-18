"""
Regression tests for a batch of small silent failures found in one audit:

- LLM extraction threw away every reply that wrapped its JSON in prose or
  a fence (paid for, then discarded as a silent failure).
- With use_llm_extraction on and no key, the same warning was logged on
  every chat turn of every session.
- OllamaProvider.chat_stream dropped temperature/top_p/stop.
- `max_completion_tokens` (OpenAI's current name) was ignored: the
  client's limit was silently replaced by the 4096 default.
- The context-window table sent GPT-4.1/GPT-5/o-series and every
  non-OpenAI, non-Claude model to a 128k default, which for the 1M-window
  models means the auto-checkpoint fires far too late.
- count_tokens memoised texts of any size; 4096 multi-megabyte entries is
  gigabytes of retained strings.
- The dashboard advertised a "Context Router — Beta" that has no
  implementation, and the graph page drew CONTESTED nodes at fallback
  opacity because the status was missing from the map.
"""
from __future__ import annotations

import logging

import pytest

from tokenmizer.api import app as app_module
from tokenmizer.graph_memory.hybrid_extractor import HybridExtractor, _parse_json_object


class TestParseJsonObject:

    def test_plain_json(self):
        assert _parse_json_object('{"goals": ["x"]}') == {"goals": ["x"]}

    def test_fenced_json(self):
        assert _parse_json_object('```json\n{"goals": ["x"]}\n```') == {"goals": ["x"]}

    def test_prose_around_json(self):
        raw = 'Sure, here is the extraction:\n{"goals": ["x"], "files": []}\nHope this helps.'
        assert _parse_json_object(raw) == {"goals": ["x"], "files": []}

    def test_no_object_is_none(self):
        assert _parse_json_object("I could not find anything.") is None
        assert _parse_json_object("") is None
        assert _parse_json_object("[1, 2]") is None

    async def test_llm_extract_survives_prose_wrapped_reply(self):
        async def provider(messages, system="", max_tokens=0):
            return {"text": 'Here you go:\n```json\n{"decisions": [{"label": "Use PostgreSQL"}]}\n```'}

        data = await HybridExtractor().llm_extract(
            [{"role": "user", "content": "we picked postgres"}], provider)
        assert data is not None
        assert data.decisions[0]["label"] == "Use PostgreSQL"


class TestExtractionWarningIsLoggedOnce:

    def test_missing_key_warns_once_not_per_call(self, monkeypatch, caplog):
        monkeypatch.setattr(app_module, "_extraction_provider", None)
        monkeypatch.setattr(app_module, "_extraction_unavailable_reason", None)
        monkeypatch.setattr(app_module.settings, "provider", "anthropic")
        monkeypatch.setattr(app_module.settings, "anthropic_api_key", "")
        with caplog.at_level(logging.WARNING, logger="tokenmizer.api.app"):
            for _ in range(5):
                assert app_module._get_extraction_provider() is None
        assert sum("no API key" in r.message for r in caplog.records) == 1


class TestOllamaStreamSampling:

    async def test_stream_payload_carries_sampling_options(self, monkeypatch):
        import httpx

        from tokenmizer.providers.providers import OllamaProvider

        captured = {}

        class _Resp:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                yield '{"message": {"content": "hi"}, "done": true}'

        class _Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def stream(self, method, url, json=None):
                captured.update(json)
                return _Resp()

        monkeypatch.setattr(httpx, "AsyncClient", _Client)
        chunks = [c async for c in OllamaProvider("llama3").chat_stream(
            [{"role": "user", "content": "x"}], temperature=0.2, top_p=0.9, stop="END")]
        assert chunks == ["hi"]
        assert captured["options"] == {
            "num_predict": 4096, "temperature": 0.2, "top_p": 0.9, "stop": ["END"]}


class TestMaxCompletionTokens:

    def test_alias_wins_over_default(self):
        req = app_module.ChatRequest(messages=[], max_completion_tokens=77)
        assert app_module._max_tokens(req) == 77

    def test_max_tokens_still_honoured(self):
        req = app_module.ChatRequest(messages=[], max_tokens=55)
        assert app_module._max_tokens(req) == 55

    def test_default_is_4096(self):
        assert app_module._max_tokens(app_module.ChatRequest(messages=[])) == 4096


class TestContextWindows:

    @pytest.mark.parametrize("model,expected", [
        ("gpt-4.1-mini", 1_000_000),
        ("gpt-4o-mini", 128_000),
        ("gpt-5", 400_000),
        ("o3-mini", 200_000),
        ("claude-sonnet-4-6", 200_000),
        ("deepseek-chat", 128_000),
        ("mistral-large-latest", 128_000),
        ("grok-3", 128_000),
        ("llama3.1:8b", 32_000),
        ("gemini-2.5-flash", 1_000_000),
        ("something-unknown", 128_000),
    ])
    def test_family_lookup(self, model, expected):
        assert app_module._context_window(model) == expected


class TestTokenCountMemoBound:

    def test_huge_text_is_not_memoised(self):
        from tokenmizer.core import tokenizer as tk

        tk.count_tokens.cache_clear()
        huge = "x " * (tk._MEMO_MAX_CHARS // 2 + 10)
        before = tk.count_tokens.cache_info().currsize
        assert tk.count_tokens(huge, "gpt-4o") > 0
        assert tk.count_tokens.cache_info().currsize == before

    def test_normal_text_is_still_memoised(self):
        from tokenmizer.core import tokenizer as tk

        tk.count_tokens.cache_clear()
        tk.count_tokens("a normal sized message", "gpt-4o")
        assert tk.count_tokens.cache_info().currsize == 1


class TestHonestSurfaces:

    def test_dashboard_does_not_advertise_a_beta_router(self):
        from tokenmizer.dashboard.page import DASHBOARD_HTML
        assert "Context Router" not in DASHBOARD_HTML
        assert ">Beta<" not in DASHBOARD_HTML
        assert "Not implemented" in DASHBOARD_HTML

    def test_every_status_has_an_opacity(self):
        from tokenmizer.graph_memory.types import NodeStatus
        from tokenmizer.graph_memory.visualization import _STATUS_OPACITY
        missing = [s.value for s in NodeStatus if s.value not in _STATUS_OPACITY]
        assert not missing, missing
