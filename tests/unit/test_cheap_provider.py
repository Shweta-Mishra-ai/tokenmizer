"""
_get_cheap_provider() — the model behind LLM extraction.

It only knew three hosted providers, all of which need a paid key, so
`provider: ollama` (local, no key) and `provider: openrouter` (which has
a free tier) could run the chat path but never LLM extraction — the
graph stayed heuristic-only with no message saying why. Extraction is a
background, structured-output task; a small open-weight model is enough
for it, and it is the difference between a sparse graph and a useful
one for anyone without a hosted-model budget.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from tokenmizer.api import app as app_module
from tokenmizer.graph_memory.graph import GraphMemory
from tokenmizer.providers.providers import OllamaProvider, OpenRouterProvider


@pytest.fixture(autouse=True)
def fresh_cheap_provider(monkeypatch):
    """_get_cheap_provider memoises into a module global; every test
    starts from an empty slot and leaves one behind."""
    monkeypatch.setattr(app_module, "_cheap_provider", None)
    monkeypatch.setattr(app_module.settings.graph_checkpoint, "extraction_model", "")
    yield
    app_module._cheap_provider = None


_KEY_FIELD = {"openrouter": "openrouter_api_key", "cohere": "cohere_api_key"}


def _configure(monkeypatch, provider: str, key: str = "", override: str = ""):
    monkeypatch.setattr(app_module.settings, "provider", provider)
    monkeypatch.setattr(app_module.settings.graph_checkpoint, "extraction_model", override)
    if provider in _KEY_FIELD:
        monkeypatch.setattr(app_module.settings, _KEY_FIELD[provider], key)


def test_ollama_needs_no_key_and_defaults_to_an_open_weight_model(monkeypatch):
    _configure(monkeypatch, "ollama")
    cheap = app_module._get_cheap_provider()
    assert isinstance(cheap, OllamaProvider)
    assert cheap.default_model == "qwen3:8b"


def test_ollama_honours_extraction_model(monkeypatch):
    _configure(monkeypatch, "ollama", override="llama3.1:8b")
    assert app_module._get_cheap_provider().default_model == "llama3.1:8b"


def test_openrouter_with_key_and_model(monkeypatch):
    _configure(monkeypatch, "openrouter", key="or-key", override="some-org/some-model:free")
    cheap = app_module._get_cheap_provider()
    assert isinstance(cheap, OpenRouterProvider)
    assert cheap.default_model == "some-org/some-model:free"


def test_openrouter_without_extraction_model_falls_back_and_says_why(monkeypatch, caplog):
    """The router's free-tier model ids rotate, so there is no id worth
    hardcoding; an operator who set a key but no model must be told what
    to set rather than silently getting heuristic extraction."""
    _configure(monkeypatch, "openrouter", key="or-key")
    with caplog.at_level(logging.WARNING):
        assert app_module._get_cheap_provider() is None
    assert any("extraction_model" in r.message for r in caplog.records)


def test_openrouter_without_key_is_none(monkeypatch):
    _configure(monkeypatch, "openrouter", override="some-org/some-model:free")
    assert app_module._get_cheap_provider() is None


def test_unknown_provider_is_still_none(monkeypatch):
    _configure(monkeypatch, "cohere", key="k")
    assert app_module._get_cheap_provider() is None


async def test_llm_extraction_runs_end_to_end_with_a_local_provider(tmp_path, monkeypatch):
    """Provider mocked at the boundary the real code uses (.chat), so
    this exercises the whole background-extraction path with
    provider: ollama and no key configured."""
    _configure(monkeypatch, "ollama")
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

    graph = GraphMemory("ollama-extract", storage_dir=str(tmp_path))
    raw = [{"role": "user", "content": "what should we store orders in, and why?"}]
    await app_module._update_graph(
        "ollama-extract", graph, raw, [dict(m) for m in raw], "qwen3:8b", {}, "orders",
    )
    for _ in range(50):
        if not app_module._background_tasks:
            break
        await asyncio.sleep(0)

    assert calls, "the local provider was never asked to extract"
    assert any("PostgreSQL" in n.label for n in graph._nodes.values())
