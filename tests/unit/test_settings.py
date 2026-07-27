"""
Regression tests for TM-06 / TM-08 (closes issue #28): config load
failures must fail CLOSED in production instead of silently falling back
to permissive defaults, and nested sub-configs must not read bare
environment variables.

Background — issue #28 as filed only covered YAML PARSE failure. Two
broader gaps found while implementing the fix:

1. `extra="ignore"` meant a MISSPELLED key in otherwise-valid YAML (e.g.
   `api_keys:` instead of `api_key:`) parsed cleanly and silently
   discarded the value — no exception, no log line, service boots with
   auth disabled. This is arguably more likely than a YAML syntax error,
   since a typo in a key name doesn't break YAML syntax at all. Fixed by
   changing to `extra="forbid"`, which raises the same way a parse
   failure already did, so it's covered by the SAME fail-closed logic.

2. Even when a YAML file loads and validates perfectly, the RESULTING
   settings could still be permissive (e.g. the operator just never set
   api_key at all — not a typo, a genuine gap). TOKENMIZER_ENV=production
   now refuses to start with an empty api_key regardless of whether
   loading itself succeeded.

3. (TM-08, same root cause as #28's "config is more fragile than it
   looks") the nested sub-config classes (CompressionSettings,
   TerseOutputSettings, etc.) subclassed BaseSettings directly with no
   env_prefix of their own, so they read BARE environment variables —
   `TerseOutputSettings()` would pick up a plain `LEVEL` or `ENABLED` env
   var from the host, generic enough names that a CI system or shell
   profile could set them by coincidence and silently reconfigure the
   product. Fixed by converting them to plain pydantic.BaseModel (they're
   nested value objects; the outer Settings already provides
   TOKENMIZER_-prefixed, __-nested env var support for them).
"""
from __future__ import annotations

import os

import pytest
import yaml

from tokenmizer.config import settings as settings_module
from tokenmizer.config.settings import (
    CompressionSettings,
    MemorySettings,
    Settings,
    TerseOutputSettings,
    get_settings,
)


@pytest.fixture(autouse=True)
def reset_settings_singleton(monkeypatch):
    """get_settings() caches into a module-level global — every test
    needs a clean slate, and must not leak TOKENMIZER_ENV into other
    tests either."""
    monkeypatch.setattr(settings_module, "_settings", None)
    monkeypatch.delenv("TOKENMIZER_ENV", raising=False)
    monkeypatch.delenv("TOKENMIZER_CONFIG", raising=False)


def _write_yaml(tmp_path, data: dict):
    path = tmp_path / "tokenmizer.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return str(path)


class TestExtraForbidCatchesTypos:

    def test_misspelled_key_raises_instead_of_silently_dropping(self, tmp_path):
        path = _write_yaml(tmp_path, {"api_keys": "super-secret-key"})  # typo
        with pytest.raises(Exception):
            Settings.from_yaml(path)

    def test_correctly_spelled_key_still_works(self, tmp_path):
        path = _write_yaml(tmp_path, {"api_key": "super-secret-key"})
        s = Settings.from_yaml(path)
        assert s.api_key == "super-secret-key"


class TestProductionFailsClosedOnLoadFailure(object):

    def test_production_raises_on_malformed_yaml(self, tmp_path, monkeypatch):
        path = tmp_path / "tokenmizer.yaml"
        path.write_text("not: valid: yaml: at: all: [", encoding="utf-8")
        monkeypatch.setenv("TOKENMIZER_ENV", "production")
        monkeypatch.setenv("TOKENMIZER_CONFIG", str(path))
        with pytest.raises(Exception):
            get_settings()

    def test_production_raises_on_misspelled_key(self, tmp_path, monkeypatch):
        path = _write_yaml(tmp_path, {"api_keys": "typo-key", "cors_origin": ["x"]})
        monkeypatch.setenv("TOKENMIZER_ENV", "production")
        monkeypatch.setenv("TOKENMIZER_CONFIG", path)
        with pytest.raises(Exception):
            get_settings()

    def test_development_still_falls_back_gracefully(self, tmp_path, monkeypatch, caplog):
        """The existing dev-mode behavior (log loudly, use defaults) must
        be preserved — this fix is about production, not about breaking
        the documented dev experience."""
        import logging
        path = tmp_path / "tokenmizer.yaml"
        path.write_text("not: valid: yaml: at: all: [", encoding="utf-8")
        monkeypatch.delenv("TOKENMIZER_ENV", raising=False)  # explicit: dev mode
        monkeypatch.setenv("TOKENMIZER_CONFIG", str(path))
        with caplog.at_level(logging.ERROR):
            s = get_settings()
        assert isinstance(s, Settings)
        assert any("Failed to load config" in r.message for r in caplog.records)


class TestProductionFailsClosedOnPermissiveResult:

    def test_production_raises_when_api_key_empty_even_if_yaml_is_valid(self, tmp_path, monkeypatch):
        """No load failure at all here — the YAML is perfectly valid, it
        just never set an api_key. That's still unsafe in production."""
        path = _write_yaml(tmp_path, {"provider": "anthropic"})  # no api_key set
        monkeypatch.setenv("TOKENMIZER_ENV", "production")
        monkeypatch.setenv("TOKENMIZER_CONFIG", path)
        with pytest.raises(Exception):
            get_settings()

    def test_production_succeeds_when_api_key_is_set(self, tmp_path, monkeypatch):
        path = _write_yaml(tmp_path, {"api_key": "real-key-value"})
        monkeypatch.setenv("TOKENMIZER_ENV", "production")
        monkeypatch.setenv("TOKENMIZER_CONFIG", path)
        s = get_settings()
        assert s.api_key == "real-key-value"

    def test_no_yaml_file_at_all_still_fails_closed_in_production(self, monkeypatch, tmp_path):
        """No tokenmizer.yaml present -> Settings() hardcoded defaults ->
        api_key is empty -> production must still refuse to start."""
        monkeypatch.setenv("TOKENMIZER_ENV", "production")
        monkeypatch.setenv("TOKENMIZER_CONFIG", str(tmp_path / "does-not-exist.yaml"))
        monkeypatch.delenv("TOKENMIZER_API_KEY", raising=False)
        with pytest.raises(Exception):
            get_settings()


class TestNestedSubconfigsDoNotReadBareEnvVars:

    def test_generic_env_var_names_are_not_picked_up(self, monkeypatch):
        monkeypatch.setenv("LEVEL", "ultra")
        monkeypatch.setenv("ENABLED", "false")
        assert TerseOutputSettings().level == "full", (
            "TerseOutputSettings picked up a bare LEVEL env var — nested "
            "sub-configs must only be settable via TOKENMIZER_ prefixed / "
            "__-nested vars on the outer Settings object"
        )
        assert TerseOutputSettings().enabled is True
        assert CompressionSettings().enabled is True
        assert MemorySettings().enabled is True

    def test_nested_delimiter_still_configures_sub_settings(self, monkeypatch):
        """The outer Settings object's env_nested_delimiter="__" path must
        still work — this fix removes BARE env var leakage, not the
        legitimate TOKENMIZER_TERSE_OUTPUT__LEVEL mechanism."""
        monkeypatch.setenv("TOKENMIZER_TERSE_OUTPUT__LEVEL", "ultra")
        s = Settings()
        assert s.terse_output.level == "ultra"
