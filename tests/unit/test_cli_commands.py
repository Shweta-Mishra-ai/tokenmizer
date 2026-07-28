"""
Regression tests for the `tokenmizer stats/checkpoint/resume` CLI
commands — previously 0% test coverage, flagged directly by the user as
an area of concern.

Three real bugs found:

1. `session_id` was interpolated raw into query strings/paths in all
   three commands — the same class of bug already fixed in the MCP
   server (see test_mcp_server.py's TestSessionIdIsUrlEncoded). A
   session_id containing reserved URL characters would produce a
   malformed or misdirected request.

2. `checkpoint` and `resume` had NO error handling around their httpx
   calls at all — unlike `stats`, which already wraps its call in a
   try/except and prints a clean "[red]Cannot reach server[/red]"
   message. An unreachable server (the single most common real-world
   failure mode for a CLI that calls a remote proxy) crashed both
   commands with a raw, unhandled httpx traceback dumped to the user's
   terminal instead of a clean error and exit code.

3. `checkpoint` and `resume` accessed response dict fields with direct
   key access (`data['checkpoint_id']`, `data["resume_context"]`)
   instead of `.get()`. A non-200, non-404 response (auth failure,
   validation error, 500) has a different body shape (FastAPI's default
   `{"detail": "..."}`), and direct key access on that raised a raw
   KeyError instead of surfacing the server's actual error message.
"""
from __future__ import annotations

import httpx
from typer.testing import CliRunner

from tokenmizer.cli import app

runner = CliRunner()


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text or str(self._json_data)

    def json(self):
        return self._json_data


class TestStatsCommand:

    def test_encodes_session_id(self, monkeypatch):
        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            return _FakeResponse(200, {"daily": {}})

        monkeypatch.setattr(httpx, "get", fake_get)
        result = runner.invoke(app, ["stats", "my session & id"])
        assert result.exit_code == 0, result.output
        assert "my session & id" not in captured["url"]

    def test_handles_unreachable_server_gracefully(self, monkeypatch):
        def fake_get(url, **kwargs):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(httpx, "get", fake_get)
        result = runner.invoke(app, ["stats"])
        assert result.exit_code == 1
        assert "Cannot reach server" in result.output


class TestCheckpointCommand:

    def test_encodes_session_id(self, monkeypatch):
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            return _FakeResponse(200, {
                "checkpoint_id": "ckpt_abc", "node_count": 3, "resume_tokens": 42,
                "resume_standard": "context here",
            })

        monkeypatch.setattr(httpx, "post", fake_post)
        result = runner.invoke(app, ["checkpoint", "my project & auth"])
        assert result.exit_code == 0, result.output
        assert "my project & auth" not in captured["url"]

    def test_handles_unreachable_server_gracefully(self, monkeypatch):
        """Previously: no try/except at all around this httpx.post call —
        an unreachable server crashed with a raw, unhandled traceback."""
        def fake_post(url, **kwargs):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(httpx, "post", fake_post)
        result = runner.invoke(app, ["checkpoint", "some-session"])
        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit), (
            f"CLI crashed with an unhandled exception instead of a clean "
            f"error: {result.exception!r}"
        )

    def test_handles_malformed_success_response_gracefully(self, monkeypatch):
        """A 200 response missing expected fields (e.g. a proxy/gateway
        returning an unexpected body shape) previously crashed with a raw
        KeyError from direct dict access."""
        def fake_post(url, **kwargs):
            return _FakeResponse(200, {"unexpected": "shape"})

        monkeypatch.setattr(httpx, "post", fake_post)
        result = runner.invoke(app, ["checkpoint", "some-session"])
        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit), (
            f"malformed success response crashed the CLI: {result.exception!r}"
        )

    def test_non_200_status_shows_server_error_text(self, monkeypatch):
        def fake_post(url, **kwargs):
            return _FakeResponse(500, text="Internal Server Error")

        monkeypatch.setattr(httpx, "post", fake_post)
        result = runner.invoke(app, ["checkpoint", "some-session"])
        assert result.exit_code == 1
        assert "Internal Server Error" in result.output


class TestResumeCommand:

    def test_encodes_session_id(self, monkeypatch):
        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            return _FakeResponse(200, {"resume_context": "ctx", "token_count": 10})

        monkeypatch.setattr(httpx, "get", fake_get)
        result = runner.invoke(app, ["resume", "session/with/slash"])
        assert result.exit_code == 0, result.output
        assert "session/with/slash" not in captured["url"]

    def test_handles_unreachable_server_gracefully(self, monkeypatch):
        def fake_get(url, **kwargs):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(httpx, "get", fake_get)
        result = runner.invoke(app, ["resume", "some-session"])
        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit), (
            f"CLI crashed with an unhandled exception instead of a clean "
            f"error: {result.exception!r}"
        )

    def test_404_still_reports_no_checkpoint_found(self, monkeypatch):
        def fake_get(url, **kwargs):
            return _FakeResponse(404)

        monkeypatch.setattr(httpx, "get", fake_get)
        result = runner.invoke(app, ["resume", "no-such-session"])
        assert result.exit_code == 1
        assert "No checkpoint found" in result.output

    def test_non_404_error_status_shows_clean_message_not_keyerror(self, monkeypatch):
        """A 401/500/etc. has a different body shape than a successful
        resume response — direct `data["resume_context"]` access on that
        previously raised a raw KeyError instead of a clean CLI error."""
        def fake_get(url, **kwargs):
            return _FakeResponse(401, {"detail": "Invalid API key"}, text="Invalid API key")

        monkeypatch.setattr(httpx, "get", fake_get)
        result = runner.invoke(app, ["resume", "some-session"])
        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit), (
            f"non-404 error response crashed the CLI: {result.exception!r}"
        )
