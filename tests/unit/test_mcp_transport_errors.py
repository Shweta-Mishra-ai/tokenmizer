"""
What an MCP client is told when the proxy is not there.

Five of the six MCP tools talk to the proxy, and the proxy is a separate
process that installing the plugin does not start. So the single most likely
outcome of a fresh install is every one of those tools answering
"[Errno 111] Connection refused" — a socket error, not a next step. It reads
as a broken plugin rather than a stopped server, and nobody files a bug for
that; they uninstall.

cli.py already solved this for the identical failure (_report_unreachable).
These tests hold the MCP path to the same standard, and keep SDK exception
text — which embeds the request URL, and with it the session id — out of the
reply.
"""
from __future__ import annotations

import httpx
import pytest

from tokenmizer.mcp import server as mcp


def test_connect_error_names_the_fix():
    message = mcp._explain(httpx.ConnectError("[Errno 111] Connection refused"))
    assert "tokenmizer serve" in message
    assert "Errno" not in message


def test_connect_error_says_which_tools_still_work():
    """analyze_file runs FileIntelligence in-process and needs no server.
    Someone told "TokenMizer is not running" should not conclude that
    everything is unavailable."""
    message = mcp._explain(httpx.ConnectError("refused"))
    assert "File analysis works without the server" in message


def test_timeout_is_distinguished_from_refusal():
    message = mcp._explain(httpx.TimeoutException("timed out"))
    assert "did not respond" in message
    assert "tokenmizer serve" not in message, (
        "a timeout means the server IS there; telling the reader to start it "
        "sends them the wrong way"
    )


@pytest.mark.parametrize("code,expected", [
    (401, "TOKENMIZER_API_KEY"),
    (403, "TOKENMIZER_API_KEY"),
    (404, "checkpointed"),
])
def test_status_errors_explain_themselves(code, expected):
    response = httpx.Response(code, request=httpx.Request("GET", "http://x/api/graph/s"))
    message = mcp._explain(httpx.HTTPStatusError("e", request=response.request,
                                                 response=response))
    assert expected in message


def test_no_session_id_leaks_through_the_error():
    """SDK exception text carries the request URL; these endpoints put the
    session id in the path and query string."""
    request = httpx.Request(
        "GET", "http://localhost:8000/api/graph/customer-acme-migration")
    response = httpx.Response(500, request=request)
    message = mcp._explain(
        httpx.HTTPStatusError("boom", request=request, response=response))
    assert "customer-acme-migration" not in message


def test_unknown_transport_failure_does_not_dump_exception_text():
    message = mcp._explain(RuntimeError("internal detail http://host/api?session_id=x"))
    assert "session_id" not in message
    assert "RuntimeError" in message


def test_handlers_surface_the_explanation_and_the_error_flag(monkeypatch):
    """The is_error flag is structural, never inferred from the text — the
    handlers' own contract."""
    monkeypatch.setattr(mcp, "_get", lambda path: {
        "error": mcp._explain(httpx.ConnectError("refused"))})

    text, is_error = mcp.handle_get_graph_stats({"session_id": "s"})
    assert is_error is True
    assert "tokenmizer serve" in text
