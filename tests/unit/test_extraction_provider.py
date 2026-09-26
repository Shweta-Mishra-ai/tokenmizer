"""
_get_extraction_provider() — the model behind LLM extraction.

Extraction used to pick a "cheap" model per provider from a hardcoded
list: three hosted providers had one, the rest (gemini, mistral, cohere,
grok, and until recently ollama/openrouter) silently got heuristic-only
extraction. That list encoded a vendor opinion the operator never asked
for, and a smaller model than the one they chose for chat — exactly the
model whose answers they had already judged good enough.

The rule is now: extraction uses the provider and model the operator
configured for chat, unless `graph_checkpoint.extraction_model` pins a
different one. Same key, same provider adapter, no second vendor list.
"""
from __future__ import annotations

import asyncio

import pytest

from tokenmizer.api import app as app_module
from tokenmizer.graph_memory.graph import GraphMemory
from tokenmizer.providers.providers import (
    AnthropicProvider,
    GeminiProvider,
    OllamaProvider,
    OpenRouterProvider,
)


@pytest.fixture(autouse=True)
def fresh_extraction_provider(monkeypatch):
    """The provider is memoised into a module global; every test starts
    from an empty slot and leaves one behind."""
    monkeypatch.setattr(app_module, "_extraction_provider", None)
    monkeypatch.setattr(app_module.settings.graph_checkpoint, "extraction_model", "")
    yield
    app_module._extraction_provider = None


_KEY_FIELD = {
    "anthropic": "anthropic_api_key", "gemini": "gemini_api_key",
    "openrouter": "openrouter_api_key", "cohere": "cohere_api_key",
}


def _configure(monkeypatch, provider: str, model: str, key: str = "", override: str = ""):
    monkeypatch.setattr(app_module.settings, "provider", provider)
    monkeypatch.setattr(app_module.settings, "default_model", model)
    monkeypatch.setattr(app_module.settings.graph_checkpoint, "extraction_model", override)
    if provider in _KEY_FIELD:
        monkeypatch.setattr(app_module.settings, _KEY_FIELD[provider], key)


def test_extraction_uses_the_chat_model_by_default(monkeypatch):
    _configure(monkeypatch, "anthropic", "claude-sonnet-4-6", key="k")
    p = app_module._get_extraction_provider()
    assert isinstance(p, AnthropicProvider)
    assert p.default_model == "claude-sonnet-4-6"


def test_extraction_model_pins_a_different_model_on_the_same_provider(monkeypatch):
    _configure(monkeypatch, "anthropic", "claude-sonnet-4-6", key="k", override="claude-haiku-4-5")
    assert app_module._get_extraction_provider().default_model == "claude-haiku-4-5"


def test_gemini_is_supported(monkeypatch):
    """Was silently heuristic-only: a Gemini key never reached extraction."""
    _configure(monkeypatch, "gemini", "gemini-2.5-flash", key="k")
    p = app_module._get_extraction_provider()
    assert isinstance(p, GeminiProvider)
    assert p.default_model == "gemini-2.5-flash"


def test_ollama_needs_no_key_and_uses_the_chat_model(monkeypatch):
    _configure(monkeypatch, "ollama", "llama3.1:8b")
    p = app_module._get_extraction_provider()
    assert isinstance(p, OllamaProvider)
    assert p.default_model == "llama3.1:8b"


def test_openrouter_uses_the_chat_model(monkeypatch):
    _configure(monkeypatch, "openrouter", "some-org/some-model", key="k")
    p = app_module._get_extraction_provider()
    assert isinstance(p, OpenRouterProvider)
    assert p.default_model == "some-org/some-model"


def test_missing_key_means_no_llm_extraction(monkeypatch, caplog):
    """A hosted provider with no key cannot extract; say so once rather
    than failing every turn in the background."""
    _configure(monkeypatch, "anthropic", "claude-sonnet-4-6", key="")
    assert app_module._get_extraction_provider() is None
    assert any("no API key" in r.message for r in caplog.records)


def test_memoised_after_first_build(monkeypatch):
    _configure(monkeypatch, "anthropic", "claude-sonnet-4-6", key="k")
    first = app_module._get_extraction_provider()
    monkeypatch.setattr(app_module.settings, "default_model", "changed")
    assert app_module._get_extraction_provider() is first


async def test_llm_extraction_runs_end_to_end(tmp_path, monkeypatch):
    """Provider mocked at the boundary the real code uses (.chat), so
    this exercises the whole background-extraction path."""
    _configure(monkeypatch, "ollama", "llama3.1:8b")
    monkeypatch.setattr(app_module.settings.graph_checkpoint, "use_llm_extraction", True)
    app_module._graph_cache.clear()
    app_module._session_locks.clear()

    class R:
        text = (
            '{"goals": [], "tasks_done": [], "tasks_wip": [], "tasks_todo": [], '
            '"decisions": [{"label": "Use PostgreSQL for order storage", '
            '"reason": "concurrent writes"}], "files": [], "errors": [], '
            '"dependencies": [], "environments": [], "endpoints": [], '
            '"schemas": [], "superseded": []}'
        )

    calls = []

    async def chat(self, **kwargs):
        calls.append(kwargs)
        return R()

    monkeypatch.setattr(OllamaProvider, "chat", chat)

    graph = GraphMemory("llm-extract", storage_dir=str(tmp_path))
    raw = [{"role": "user", "content": "what should we store orders in, and why?"}]
    await app_module._update_graph(
        "llm-extract", graph, raw, [dict(m) for m in raw], "llama3.1:8b", {}, "orders",
    )
    # Await the task rather than spinning on sleep(0): the extraction pass
    # now runs in a worker thread (app._extract_heuristic — it was stalling
    # the event loop for as long as it ran), so yielding to the loop no
    # longer implies the work is done. A fixed number of yields was always
    # a guess about scheduling.
    await asyncio.gather(*list(app_module._background_tasks),
                         return_exceptions=True)

    assert calls, "the provider was never asked to extract"
    assert any("PostgreSQL" in n.label for n in graph._nodes.values())
