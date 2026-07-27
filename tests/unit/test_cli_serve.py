"""
Regression tests for the `tokenmizer serve` CLI command, part of TM-06's
scope: making the bind-address defaults consistent, config-driven, and
safe by default.

Background — two separate bugs found while fixing TM-06:

1. `tokenmizer.yaml` ships with `proxy_host: 0.0.0.0` / `proxy_port: 8000`
   documented as configuration, and `Settings.proxy_host`/`proxy_port`
   exist on the Settings model — but nothing in the codebase ever read
   them. `cli.py`'s `serve()` command had its own, entirely independent
   hardcoded typer defaults (`"0.0.0.0"`, `8000`). Editing
   tokenmizer.yaml's proxy_host/proxy_port did nothing at all — the exact
   "looks like it does something but doesn't" pattern this audit is
   about. Fixed: `serve()`'s host/port options now default to None and
   fall back to `get_settings().proxy_host` / `.proxy_port` when the
   caller doesn't pass an explicit flag, so the config file is no longer
   silently ignored.

2. `Settings.proxy_host` defaulted to `0.0.0.0` (bind all interfaces).
   Now that it's actually wired in, that default matters: changed to
   `127.0.0.1`. This does not affect the documented Docker deployment
   path, which always passes `--host 0.0.0.0 --port 8000` explicitly in
   the Dockerfile's CMD — an explicit CLI flag always overrides the
   config-driven default.
"""
from __future__ import annotations

from typer.testing import CliRunner

from tokenmizer.cli import app
from tokenmizer.config import settings as settings_module

runner = CliRunner()


def _fresh_settings(monkeypatch, **overrides):
    """Reset the get_settings() singleton and apply overrides, so each
    test starts from a known Settings() rather than whatever an earlier
    test's yaml-loading side effects left behind."""
    monkeypatch.setattr(settings_module, "_settings", None)
    s = settings_module.Settings(**overrides)
    monkeypatch.setattr(settings_module, "_settings", s)
    return s


class TestServeUsesConfigDrivenDefaults:

    def test_no_flags_uses_settings_proxy_host_and_port(self, monkeypatch):
        _fresh_settings(monkeypatch, proxy_host="10.0.0.5", proxy_port=9001)

        captured = {}

        def _fake_run(app_path, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr("uvicorn.run", _fake_run)

        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0, result.output
        assert captured.get("host") == "10.0.0.5"
        assert captured.get("port") == 9001

    def test_explicit_flag_overrides_config(self, monkeypatch):
        """An explicit --host/--port must always win over the config file
        — this is what keeps the Dockerfile's hardcoded
        `--host 0.0.0.0 --port 8000` working unchanged."""
        _fresh_settings(monkeypatch, proxy_host="127.0.0.1", proxy_port=8000)

        captured = {}

        def _fake_run(app_path, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr("uvicorn.run", _fake_run)

        result = runner.invoke(app, ["serve", "--host", "0.0.0.0", "--port", "9999"])
        assert result.exit_code == 0, result.output
        assert captured.get("host") == "0.0.0.0"
        assert captured.get("port") == 9999


class TestDefaultBindAddressIsSafe:

    def test_settings_default_proxy_host_is_localhost(self):
        from tokenmizer.config.settings import Settings
        assert Settings().proxy_host == "127.0.0.1", (
            "default bind address should be localhost, not all interfaces "
            "(0.0.0.0) — an operator who runs `tokenmizer serve` bare "
            "should not be exposed on every network interface by default"
        )
