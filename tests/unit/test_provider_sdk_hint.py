"""A missing provider SDK is reported where the operator will see it.

A plain `pip install tokenmizer` has no provider SDK (they are extras), and the
adapters import theirs on the first request. The first chat call therefore
failed with an opaque 502, and the actual cause, "pip install anthropic", was
only in the server log. The proxy now says so at startup, in /health, and in
the 502 itself. /health does not turn "degraded" for it: a deployment that only
checkpoints through MCP never needs a provider.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tokenmizer.api import app as app_module
from tokenmizer.api.app import app
from tokenmizer.providers import providers
from tokenmizer.providers.providers import _PROVIDER_SDK, missing_provider_sdk
from tokenmizer.security.ownership import OwnershipStore

HINT = 'pip install "tokenmizer[anthropic]"'
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.settings, "api_key", "", raising=False)
    monkeypatch.setattr(app_module.settings.graph_checkpoint, "storage_dir", str(tmp_path))
    monkeypatch.setattr(app_module, "_ownership", OwnershipStore(storage_dir=str(tmp_path)))
    app_module._graph_cache.clear()
    with TestClient(app) as c:
        yield c


class TestTheHelper:

    def test_a_present_sdk_is_not_reported(self, monkeypatch):
        monkeypatch.setattr(providers.importlib.util, "find_spec", lambda name: object())
        assert missing_provider_sdk("anthropic") is None

    @pytest.mark.parametrize("provider,extra", [
        ("anthropic", "anthropic"), ("claude", "anthropic"),
        ("openai", "openai"), ("deepseek", "openai"), ("mistral", "openai"),
        ("grok", "openai"), ("openrouter", "openai"),
        ("gemini", "gemini"), ("cohere", "cohere"),
    ])
    def test_a_missing_sdk_names_the_extra_that_installs_it(self, monkeypatch, provider, extra):
        monkeypatch.setattr(providers.importlib.util, "find_spec", lambda name: None)
        assert missing_provider_sdk(provider) == f'pip install "tokenmizer[{extra}]"'

    def test_a_missing_parent_package_counts_as_missing(self, monkeypatch):
        """find_spec("google.genai") raises, rather than returning None, when
        `google` itself is not installed."""
        def raises(name):
            raise ModuleNotFoundError(name)
        monkeypatch.setattr(providers.importlib.util, "find_spec", raises)
        assert missing_provider_sdk("gemini") == 'pip install "tokenmizer[gemini]"'

    @pytest.mark.parametrize("provider", ["ollama", "", None, "no-such-provider"])
    def test_providers_without_an_sdk_are_never_reported(self, monkeypatch, provider):
        monkeypatch.setattr(providers.importlib.util, "find_spec", lambda name: None)
        assert missing_provider_sdk(provider) is None

    def test_the_provider_name_is_case_insensitive(self, monkeypatch):
        monkeypatch.setattr(providers.importlib.util, "find_spec", lambda name: None)
        assert missing_provider_sdk("Anthropic") == HINT

    def test_every_hint_names_an_extra_that_exists(self):
        """The command must be installable: each extra is declared in pyproject."""
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        for _module, extra in set(_PROVIDER_SDK.values()):
            assert f"\n{extra} = [" in pyproject, f"pyproject has no [{extra}] extra"


class TestWhereItIsReported:

    def test_health_names_the_hint_without_changing_the_status(self, client, monkeypatch):
        """Compared against the same server without the hint, not against an
        absolute "ok": other tests in the run leave failure counters in the
        shared analytics object, which legitimately make /health degraded."""
        monkeypatch.setattr(app_module, "missing_provider_sdk", lambda provider: None)
        baseline = client.get("/health").json()["status"]

        monkeypatch.setattr(app_module, "missing_provider_sdk", lambda provider: HINT)
        body = client.get("/health").json()

        assert body["provider_sdk_missing"] == HINT
        assert body["status"] == baseline, "a missing SDK must not change /health's status"

    def test_health_is_null_when_nothing_is_missing(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "missing_provider_sdk", lambda provider: None)
        assert client.get("/health").json()["provider_sdk_missing"] is None

    def test_startup_warns_with_the_command(self, monkeypatch, tmp_path, caplog):
        monkeypatch.setattr(app_module, "missing_provider_sdk", lambda provider: HINT)
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "storage_dir", str(tmp_path))
        with caplog.at_level(logging.WARNING, logger="tokenmizer.api.app"):
            with TestClient(app):
                pass
        assert any(HINT in r.getMessage() and "SDK is not installed" in r.getMessage()
                   for r in caplog.records)

    @staticmethod
    def _failing_chat(client, monkeypatch, hint):
        class Broken:
            async def chat(self, **kwargs):
                raise RuntimeError("upstream exploded: https://internal.example/secret-path")

        monkeypatch.setattr(app_module, "_provider", Broken())
        monkeypatch.setattr(app_module, "missing_provider_sdk", lambda provider: hint)
        return client.post("/v1/chat/completions", json={
            "session_id": "sdk-hint", "messages": [{"role": "user", "content": "hello"}]})

    def test_the_502_names_the_cause_when_the_sdk_is_missing(self, client, monkeypatch):
        r = self._failing_chat(client, monkeypatch, HINT)
        assert r.status_code == 502
        assert HINT in r.json()["detail"]
        assert "ref:" in r.json()["detail"]

    def test_the_502_stays_opaque_otherwise(self, client, monkeypatch):
        r = self._failing_chat(client, monkeypatch, None)
        assert r.status_code == 502
        assert "SDK" not in r.json()["detail"]
        assert "internal.example" not in r.text and "exploded" not in r.text
