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

import re

import httpx
import pytest
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
        # Asserts the reader is told what to DO, not the exact prose. The
        # previous wording was "Cannot reach server: [Errno 111]
        # Connection refused", which is what the stack knows rather than
        # what the reader needs.
        assert "tokenmizer serve" in _plain(result.output), result.output


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


class TestAnalyzeCommand:
    """`tokenmizer analyze` and POST /api/analyze close a gap the README
    previously documented as missing: file analysis outside Claude Code."""

    def test_analyzes_a_csv(self, tmp_path):
        from typer.testing import CliRunner

        from tokenmizer.cli import app as cli_app

        f = tmp_path / "data.csv"
        f.write_text("region,revenue\n" + "\n".join(f"EMEA,{i}" for i in range(300)))

        res = CliRunner().invoke(cli_app, ["analyze", str(f), "--token-budget", "200"])
        assert res.exit_code == 0, res.output
        assert "data.csv" in res.output
        assert "300 rows" in res.output or "region" in res.output

    def test_raw_mode_prints_only_the_summary(self, tmp_path):
        from typer.testing import CliRunner

        from tokenmizer.cli import app as cli_app

        f = tmp_path / "d.csv"
        f.write_text("a,b\n1,2\n3,4\n")
        res = CliRunner().invoke(cli_app, ["analyze", str(f), "--raw"])
        assert res.exit_code == 0
        assert "tokens ->" not in res.output, "raw mode must omit the stats header"

    # `kind` picks the path at run time from `tmp_path`, rather than
    # hardcoding one. The directory case used to pass `/tmp`, which does
    # not exist on Windows — so the CLI correctly answered "file not
    # found" and the test, which expected "not a file", failed for a
    # reason that had nothing to do with the behaviour being checked.
    @pytest.mark.parametrize("kind,expect", [
        ("missing", "not found"),
        ("directory", "not a file"),
    ])
    def test_bad_input_exits_nonzero_with_a_reason(self, kind, expect, tmp_path):
        from typer.testing import CliRunner

        from tokenmizer.cli import app as cli_app

        target = tmp_path / "nope.csv" if kind == "missing" else tmp_path
        res = CliRunner().invoke(cli_app, ["analyze", str(target)])
        assert res.exit_code == 1
        assert expect in res.output.lower()

    def test_negative_budget_is_rejected(self, tmp_path):
        from typer.testing import CliRunner

        from tokenmizer.cli import app as cli_app

        f = tmp_path / "d.csv"
        f.write_text("a,b\n1,2\n")
        res = CliRunner().invoke(cli_app, ["analyze", str(f), "--token-budget", "0"])
        assert res.exit_code == 1


class TestAnalyzeEndpoint:
    def test_returns_a_budgeted_summary(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        import tokenmizer.api.app as app_module
        from tokenmizer.security.ownership import OwnershipStore

        monkeypatch.setattr(app_module, "_ownership", OwnershipStore(storage_dir=str(tmp_path)))
        with TestClient(app_module.app) as c:
            r = c.post("/api/analyze", json={
                "filename": "sales.csv",
                "content": "region,revenue\n" + "\n".join(f"EMEA,{i}" for i in range(400)),
                "token_budget": 250,
            })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["file_type"] == "csv"
        assert body["extracted_tokens"] <= 250, "the budget is the contract"
        assert body["original_tokens"] > body["extracted_tokens"]

    @pytest.mark.parametrize("payload", [
        {"filename": "x.csv", "content": "a,b\n1,2\n", "token_budget": 0},
        {"filename": "x.csv", "content": "a,b\n1,2\n", "token_budget": 10 ** 7},
        {"filename": "  ", "content": "a,b\n1,2\n"},
    ])
    def test_invalid_input_is_422(self, tmp_path, monkeypatch, payload):
        from fastapi.testclient import TestClient

        import tokenmizer.api.app as app_module
        from tokenmizer.security.ownership import OwnershipStore

        monkeypatch.setattr(app_module, "_ownership", OwnershipStore(storage_dir=str(tmp_path)))
        with TestClient(app_module.app) as c:
            assert c.post("/api/analyze", json=payload).status_code == 422

    def test_does_not_accept_a_server_side_path(self, tmp_path, monkeypatch):
        """Content is sent inline on purpose — accepting a path would be
        an arbitrary-file-read primitive against the server."""
        from fastapi.testclient import TestClient

        import tokenmizer.api.app as app_module
        from tokenmizer.security.ownership import OwnershipStore

        monkeypatch.setattr(app_module, "_ownership", OwnershipStore(storage_dir=str(tmp_path)))
        with TestClient(app_module.app) as c:
            r = c.post("/api/analyze", json={"file_path": "/etc/passwd"})
        assert r.status_code == 422


def _plain(text: str) -> str:
    """Strip ANSI styling before asserting on CLI output.

    Rich styles each run separately, so a colourized `--server` arrives as
    `ESC[1;36m-ESC[0mESC[1;36m-serverESC[0m` and the literal substring is
    not present. Whether it colourizes depends on terminal detection, so a
    test that asserts on raw output passes locally and fails in CI.
    """
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class TestFirstRunOutput:
    """The first two commands a new reader runs are `analyze` and
    `stats`. Both printed something that reads as a broken program."""

    def test_a_digest_larger_than_its_source_is_not_reported_as_negative(self, tmp_path):
        """A three-row CSV becomes a schema, types and statistics — bigger
        than the source, and correct. The header said "-536% smaller"."""
        from typer.testing import CliRunner

        from tokenmizer.cli import app as cli_app

        f = tmp_path / "tiny.csv"
        f.write_text("id,name,amount\n1,alpha,10\n2,beta,20\n3,gamma,30\n")
        res = CliRunner().invoke(cli_app, ["analyze", str(f)])
        out = _plain(res.output)
        assert res.exit_code == 0
        assert "-536" not in out and "% smaller" not in out, out
        assert "larger" in out, out

    @pytest.mark.parametrize("command", [
        ["stats"],
        ["checkpoint", "sess"],
        ["resume", "sess"],
    ])
    def test_an_unreachable_server_says_what_to_do(self, command):
        """"[Errno 111] Connection refused" is what the stack knows, not
        what the reader needs — which is `tokenmizer serve`."""
        from typer.testing import CliRunner

        from tokenmizer.cli import app as cli_app

        res = CliRunner().invoke(
            cli_app, command + ["--server", "http://127.0.0.1:9"])
        out = _plain(res.output)
        assert res.exit_code == 1
        assert "tokenmizer serve" in out, out
        assert "Errno" not in out, out

    def test_the_suggested_flag_actually_exists(self):
        """The message tells the reader to pass `--server`. It nearly told
        them to set an environment variable that does not exist."""
        from typer.testing import CliRunner

        from tokenmizer.cli import app as cli_app

        out = _plain(CliRunner().invoke(cli_app, ["stats", "--help"]).output)
        assert "--server" in out, out
