"""Local extraction through Ollama.

A small HTTP server stands in for Ollama (/api/tags, /api/chat), so the real
adapter, the real HTTP calls and the real extraction parse all run. What it
cannot show is how good any actual model is — that is what
benchmarks/extraction_models measures on a machine with Ollama.
"""
from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tokenmizer.api import app as app_module
from tokenmizer.providers.ollama_models import is_thinking_model, pick_model

REPLY = {
    "goals": [], "tasks_done": ["Added rate limiting to the login route"],
    "tasks_wip": [], "tasks_todo": ["Write tests for the refund flow"],
    "decisions": [{"label": "Use PostgreSQL for orders", "reason": "row locks"}],
    "files": ["src/api/orders.py"], "errors": [], "dependencies": [],
    "environments": [], "endpoints": [], "schemas": [], "superseded": [],
}


class _FakeOllama:
    def __init__(self, models, reject_format=False):
        self.models = models
        self.reject_format = reject_format
        self.chats: list[dict] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": m} for m in outer.models]})
                else:
                    self._send(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.chats.append(body)
                if outer.reject_format and "format" in body:
                    self._send(400, {"error": "unknown field format"})
                    return
                self._send(200, {"message": {"role": "assistant", "content": json.dumps(REPLY)},
                                 "done": True, "done_reason": "stop",
                                 "prompt_eval_count": 100, "eval_count": 50})

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def ollama():
    servers = []

    def start(models, **kw):
        s = _FakeOllama(models, **kw)
        servers.append(s)
        return s
    yield start
    for s in servers:
        s.close()


@pytest.fixture
def extraction_settings(monkeypatch):
    gc = app_module.settings.graph_checkpoint
    monkeypatch.setattr(gc, "use_llm_extraction", True)
    monkeypatch.setattr(gc, "extraction_provider", "ollama")
    monkeypatch.setattr(gc, "extraction_model", "auto")
    monkeypatch.setattr(app_module, "_extraction_provider", None)
    monkeypatch.setattr(app_module, "_extraction_unavailable_reason", None)
    monkeypatch.setattr(app_module, "_ollama_retry_at", 0.0)
    return gc


# ── choosing the model ───────────────────────────────────────────────────────

def test_auto_takes_the_highest_ranked_installed_model():
    model, why = pick_model(["llama3.2:3b", "qwen2.5:7b-instruct-q4_K_M", "nomic-embed-text"])
    assert model == "qwen2.5:7b-instruct-q4_K_M" and "auto" in why


def test_embedding_models_are_never_picked():
    model, why = pick_model(["nomic-embed-text:latest", "mxbai-embed-large"])
    assert model is None and "ollama pull" in why


def test_an_unranked_model_is_used_with_advice():
    model, why = pick_model(["mistral:7b"])
    assert model == "mistral:7b" and "ollama pull" in why


def test_a_pinned_model_that_is_not_installed_is_not_swapped_for_another():
    model, why = pick_model(["qwen2.5:7b"], "llama3.1:8b")
    assert model is None and "ollama pull llama3.1:8b" in why


@pytest.mark.parametrize("name,thinks", [("qwen3:8b", True), ("deepseek-r1:7b", True),
                                          ("gpt-oss:20b", True), ("qwen2.5:7b", False)])
def test_thinking_families(name, thinks):
    assert is_thinking_model(name) is thinks


# ── the call ─────────────────────────────────────────────────────────────────

def test_extraction_runs_locally_with_schema_and_context(ollama, extraction_settings, monkeypatch):
    server = ollama(["llama3.2:3b", "qwen3:8b"])
    monkeypatch.setattr(app_module.settings, "ollama_base_url", server.url)
    provider = app_module._get_extraction_provider()
    assert provider is not None and provider.default_model == "qwen3:8b"

    from tokenmizer.graph_memory.hybrid_extractor import HybridExtractor
    from tokenmizer.graph_memory.llm_extraction import extraction_fn
    data = asyncio.run(HybridExtractor().llm_extract(
        [{"role": "user", "content": "We'll use PostgreSQL for orders."}], extraction_fn(provider)))
    assert data is not None and data.files == ["src/api/orders.py"]

    sent = server.chats[-1]
    assert sent["model"] == "qwen3:8b"
    assert sent["format"]["type"] == "object"            # constrained to the schema
    assert sent["think"] is False                         # thinking off for a reasoning model
    assert sent["options"]["temperature"] == 0
    assert sent["options"]["num_ctx"] == app_module.settings.ollama_num_ctx


def test_an_older_ollama_without_schema_support_is_asked_again(ollama, extraction_settings, monkeypatch):
    server = ollama(["qwen2.5:7b"], reject_format=True)
    monkeypatch.setattr(app_module.settings, "ollama_base_url", server.url)
    from tokenmizer.graph_memory.hybrid_extractor import HybridExtractor
    from tokenmizer.graph_memory.llm_extraction import extraction_fn
    provider = app_module._get_extraction_provider()
    data = asyncio.run(HybridExtractor().llm_extract(
        [{"role": "user", "content": "x"}], extraction_fn(provider)))
    assert data is not None
    assert "format" in server.chats[0] and "format" not in server.chats[1]


def test_a_stopped_ollama_costs_one_check_a_minute(extraction_settings, monkeypatch):
    monkeypatch.setattr(app_module.settings, "ollama_base_url", "http://127.0.0.1:9")
    calls = []
    import tokenmizer.providers.ollama_models as om
    real = om.installed_models
    monkeypatch.setattr(om, "installed_models", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert app_module._get_extraction_provider() is None
    assert app_module._get_extraction_provider() is None
    assert len(calls) == 1


def test_chat_on_ollama_keeps_extracting_with_the_chat_model(monkeypatch):
    """Unchanged behaviour: no extraction_provider set, chat on Ollama."""
    gc = app_module.settings.graph_checkpoint
    monkeypatch.setattr(gc, "extraction_provider", "")
    monkeypatch.setattr(gc, "extraction_model", "")
    monkeypatch.setattr(app_module.settings, "provider", "ollama")
    monkeypatch.setattr(app_module.settings, "default_model", "llama3.1:8b")
    monkeypatch.setattr(app_module, "_extraction_provider", None)
    monkeypatch.setattr(app_module, "_ollama_retry_at", 0.0)
    assert app_module._get_extraction_provider().default_model == "llama3.1:8b"


def test_local_only_refuses_a_remote_extraction_provider():
    from tokenmizer.config.settings import Settings
    from tokenmizer.core.privacy import PrivacyConfigError, validate
    s = Settings(provider="ollama", privacy={"local_only": True},
                 graph_checkpoint={"use_llm_extraction": True, "extraction_provider": "openai"})
    with pytest.raises(PrivacyConfigError, match="extraction_provider"):
        validate(s)
    validate(Settings(provider="ollama", privacy={"local_only": True},
                      graph_checkpoint={"use_llm_extraction": True,
                                        "extraction_provider": "ollama"}))


def test_the_benchmark_ranks_installed_models(ollama, monkeypatch, capsys, tmp_path):
    server = ollama(["qwen2.5:7b", "nomic-embed-text"])
    import sys

    from benchmarks.extraction_models.__main__ import main
    out = tmp_path / "r.json"
    monkeypatch.setattr(sys, "argv", ["x", "--base-url", server.url, "--json", str(out)])
    assert main() == 0
    rows = json.loads(out.read_text())
    assert [r["model"] for r in rows] == ["(heuristic only)", "qwen2.5:7b"]   # embedder skipped
    assert rows[1]["calls"] > 0 and rows[1]["json_failures"] == 0
    assert "macro F1" in capsys.readouterr().out
