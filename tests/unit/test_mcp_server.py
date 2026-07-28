"""
Regression tests for the MCP stdio server.

Invariants under test:
  1. isError is structural — validation failures and handler crashes are
     reported with isError: true regardless of message text.
  2. No input terminates the read loop: malformed JSON returns -32700,
     non-object messages return -32600, handler exceptions return -32603,
     and subsequent requests are still served.
  3. Required arguments are validated with typed, descriptive errors.
"""
import io
import json
import sys

from tokenmizer.mcp import server as mcp

# ── handle_tool_call: structural isError ─────────────────────────────────────

def test_missing_required_arg_is_error():
    text, is_error = mcp.handle_tool_call("checkpoint_session", {})
    assert is_error is True
    assert "session_id" in text


def test_missing_arg_resume_is_error():
    text, is_error = mcp.handle_tool_call("resume_session", {})
    assert is_error is True
    assert "session_id" in text


def test_invalid_level_is_error():
    text, is_error = mcp.handle_tool_call(
        "resume_session", {"session_id": "s1", "level": "verbose"})
    assert is_error is True
    assert "level" in text


def test_unknown_tool_is_error():
    text, is_error = mcp.handle_tool_call("no_such_tool", {})
    assert is_error is True
    assert "no_such_tool" in text


def test_non_dict_arguments_is_error():
    text, is_error = mcp.handle_tool_call("checkpoint_session", [1, 2])
    assert is_error is True
    assert "JSON object" in text


def test_bad_token_budget_is_error():
    text, is_error = mcp.handle_tool_call(
        "analyze_file", {"file_path": "x.csv", "token_budget": "500"})
    assert is_error is True
    assert "token_budget" in text


def test_bool_token_budget_rejected():
    # bool is an int subclass — must not sneak through the isinstance check
    text, is_error = mcp.handle_tool_call(
        "analyze_file", {"file_path": "x.csv", "token_budget": True})
    assert is_error is True


def test_file_not_found_is_error(tmp_path):
    text, is_error = mcp.handle_tool_call(
        "analyze_file", {"file_path": str(tmp_path / "nope.csv")})
    assert is_error is True
    assert "not found" in text.lower()


def test_analyze_file_success_not_error(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("region,revenue\nEMEA,100\nAPAC,200\n")
    text, is_error = mcp.handle_tool_call("analyze_file", {"file_path": str(f)})
    assert is_error is False
    assert "File Analysis" in text


def test_handler_crash_is_error_not_exception(monkeypatch):
    def failing_handler(args):
        raise RuntimeError("simulated handler failure")
    # handle_tool_call builds its dispatch dict from module globals at call
    # time, so patching the module attribute is picked up.
    monkeypatch.setattr(mcp, "handle_get_savings_stats", failing_handler)
    text, is_error = mcp.handle_tool_call("get_savings_stats", {})
    assert is_error is True
    assert "simulated handler failure" in text or "internal error" in text.lower()


# ── stdio transport: survives hostile input ──────────────────────────────────

def _run_lines(monkeypatch, lines: list[str]) -> list[dict]:
    """Feed lines to run_stdio_server, return parsed JSON responses."""
    stdin = io.StringIO("\n".join(lines) + "\n")
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    mcp.run_stdio_server()
    return [json.loads(ln) for ln in stdout.getvalue().splitlines() if ln.strip()]


def test_stdio_initialize_handshake(monkeypatch):
    out = _run_lines(monkeypatch, [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
    ])
    assert out[0]["id"] == 1
    assert out[0]["result"]["protocolVersion"] == "2024-11-05"
    assert out[0]["result"]["serverInfo"]["name"] == "tokenmizer"


def test_stdio_survives_malformed_json(monkeypatch):
    """Malformed line → parse error response, next request still served."""
    out = _run_lines(monkeypatch, [
        "{this is not json",
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    ])
    assert out[0]["error"]["code"] == -32700
    assert out[1]["id"] == 2
    assert len(out[1]["result"]["tools"]) == 6


def test_stdio_survives_non_object_json(monkeypatch):
    """[1,2] is valid JSON but not a request object — was a server-killer."""
    out = _run_lines(monkeypatch, [
        "[1, 2]",
        '"just a string"',
        json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}),
    ])
    assert out[0]["error"]["code"] == -32600
    assert out[1]["error"]["code"] == -32600
    assert out[2]["id"] == 3


def test_stdio_handler_exception_yields_jsonrpc_error(monkeypatch):
    """A crash inside request handling → -32603 response, loop survives."""
    def boom(req, send):
        raise RuntimeError("handler exploded")
    real = mcp._handle_request
    calls = {"n": 0}

    def flaky(req, send):
        calls["n"] += 1
        if calls["n"] == 1:
            return boom(req, send)
        return real(req, send)

    monkeypatch.setattr(mcp, "_handle_request", flaky)
    out = _run_lines(monkeypatch, [
        json.dumps({"jsonrpc": "2.0", "id": 4, "method": "tools/list"}),
        json.dumps({"jsonrpc": "2.0", "id": 5, "method": "tools/list"}),
    ])
    assert out[0]["error"]["code"] == -32603
    assert "handler exploded" in out[0]["error"]["message"]
    assert out[1]["id"] == 5 and "result" in out[1]


def test_stdio_missing_arg_reports_is_error_true(monkeypatch):
    """A missing required argument must reach the client as isError: true."""
    out = _run_lines(monkeypatch, [
        json.dumps({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                    "params": {"name": "checkpoint_session", "arguments": {}}}),
    ])
    assert out[0]["result"]["isError"] is True
    assert "session_id" in out[0]["result"]["content"][0]["text"]


def test_stdio_unknown_method_error(monkeypatch):
    out = _run_lines(monkeypatch, [
        json.dumps({"jsonrpc": "2.0", "id": 7, "method": "bogus/method"}),
    ])
    assert out[0]["error"]["code"] == -32601


def test_stdio_notifications_get_no_response(monkeypatch):
    out = _run_lines(monkeypatch, [
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps({"jsonrpc": "2.0", "id": 8, "method": "tools/list"}),
    ])
    assert len(out) == 1
    assert out[0]["id"] == 8


def test_tool_schemas_are_valid():
    """Every tool: name, description, well-formed object inputSchema."""
    for tool in mcp.TOOLS:
        assert tool["name"]
        assert tool["description"]
        schema = tool["inputSchema"]
        assert schema["type"] == "object"
        assert isinstance(schema.get("properties", {}), dict)
        for req_key in schema.get("required", []):
            assert req_key in schema["properties"], (
                f"{tool['name']}: required key {req_key} not in properties"
            )


class TestSessionIdIsUrlEncoded:
    """Regression test for TM-37: session_id was interpolated raw into
    query strings/paths in several tool handlers (only the `query` param
    in why_decision was properly quote()'d). A session_id containing
    reserved URL characters (space, '&', '#', '/') would produce a
    malformed request or silently truncate/misdirect it."""

    def _capture_get_url(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(mcp, "_get", lambda path: (captured.setdefault("path", path), {"ok": True})[1])
        return captured

    def _capture_post_url(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(mcp, "_post", lambda path, body: (captured.setdefault("path", path), {"ok": True})[1])
        return captured

    def test_checkpoint_session_encodes_session_id(self, monkeypatch):
        captured = self._capture_post_url(monkeypatch)
        mcp.handle_checkpoint_session({"session_id": "my project & auth"})
        assert "my project & auth" not in captured["path"]
        assert "%26" in captured["path"] or "+" in captured["path"] or "%20" in captured["path"]

    def test_resume_session_encodes_session_id(self, monkeypatch):
        captured = self._capture_get_url(monkeypatch)
        mcp.handle_resume_session({"session_id": "session/with/slashes"})
        assert "session/with/slashes" not in captured["path"]

    def test_get_graph_stats_encodes_session_id(self, monkeypatch):
        captured = self._capture_get_url(monkeypatch)
        mcp.handle_get_graph_stats({"session_id": "session#fragment"})
        assert "session#fragment" not in captured["path"]
