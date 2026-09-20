"""
The MCP tools must work when the proxy is not running.

The MCP server ships with the plugin; `tokenmizer serve` is a separate
process nobody started. So the most likely state on a fresh install is
"not running", and five of the six tools answered nothing but an
explanation of that — for a graph that was sitting on disk the whole
time. They now read and write the same SQLite store the proxy uses.

Only a TRANSPORT failure falls back. A 401/403/404 means the proxy is
there and said no, and answering from local storage would walk straight
past the session-ownership boundary it was enforcing.
"""
from __future__ import annotations

import httpx
import pytest

from tokenmizer.graph_memory.graph import GraphMemory
from tokenmizer.mcp import server as mcp

TRANSCRIPT = [
    {"role": "user", "content": "We are building the checkout service"},
    {"role": "assistant", "content":
     "Decided: PostgreSQL for order storage. Created internal/store/postgres.go."},
    {"role": "assistant", "content": "Switched from moment.js to date-fns."},
]


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """No proxy at all: every HTTP call raises a connect error."""
    def _refuse(*a, **kw):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", _refuse)
    monkeypatch.setattr(httpx, "post", _refuse)
    monkeypatch.setattr(mcp, "TOKENMIZER_STORAGE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def denied(monkeypatch, tmp_path):
    """A proxy that is running and refusing: must NOT fall back."""
    class _Resp:
        status_code = 403

        def raise_for_status(self):
            raise httpx.HTTPStatusError("forbidden", request=None, response=self)

        def json(self):
            return {}

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _Resp())
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: _Resp())
    monkeypatch.setattr(mcp, "TOKENMIZER_STORAGE_DIR", str(tmp_path))
    return tmp_path


class TestOfflineFallback:

    def test_checkpoint_writes_the_graph_without_the_proxy(self, offline):
        text, is_error = mcp.handle_checkpoint_session(
            {"session_id": "offline-1", "messages": TRANSCRIPT})
        assert not is_error, text
        assert "saved to graph memory" in text
        assert "local graph memory" in text
        g = GraphMemory("offline-1", storage_dir=str(offline))
        assert len(g._nodes) > 0, "nothing reached the store"

    def test_resume_reads_the_graph_without_the_proxy(self, offline):
        mcp.handle_checkpoint_session({"session_id": "offline-2", "messages": TRANSCRIPT})
        text, is_error = mcp.handle_resume_session({"session_id": "offline-2"})
        assert not is_error, text
        assert "PostgreSQL" in text or "Postgres" in text
        assert "local graph memory" in text

    def test_graph_stats_read_the_store(self, offline):
        mcp.handle_checkpoint_session({"session_id": "offline-3", "messages": TRANSCRIPT})
        text, is_error = mcp.handle_get_graph_stats({"session_id": "offline-3"})
        assert not is_error, text
        assert "Total nodes:" in text
        assert "local graph memory" in text

    def test_why_walks_the_chain_from_the_store(self, offline):
        mcp.handle_checkpoint_session({"session_id": "offline-4", "messages": TRANSCRIPT})
        text, is_error = mcp.handle_why_decision(
            {"session_id": "offline-4", "query": "date-fns"})
        assert not is_error, text
        assert "moment" in text.lower(), text

    def test_resume_of_an_unknown_session_is_not_an_error(self, offline):
        text, is_error = mcp.handle_resume_session({"session_id": "never-seen"})
        assert not is_error
        assert "Nothing to resume" in text

    def test_savings_says_why_there_is_nothing_rather_than_reporting_zero(self, offline):
        text, is_error = mcp.handle_get_savings_stats({})
        assert not is_error
        assert "not running" in text
        assert "0 tokens saved" not in text


class TestARunningProxyIsNeverBypassed:

    def test_a_refusal_is_not_answered_from_local_storage(self, denied):
        """403 means the proxy is enforcing ownership. Falling back would
        hand the caller a session it had just been refused."""
        g = GraphMemory("denied-1", storage_dir=str(denied))
        g.add_node(*_decision())
        g._persist(force=True)

        text, is_error = mcp.handle_resume_session({"session_id": "denied-1"})
        assert is_error, text
        assert "local graph memory" not in text

    def test_a_refused_checkpoint_does_not_write_locally(self, denied):
        text, is_error = mcp.handle_checkpoint_session(
            {"session_id": "denied-2", "messages": TRANSCRIPT})
        assert is_error, text
        g = GraphMemory("denied-2", storage_dir=str(denied))
        assert len(g._nodes) == 0, "a refused checkpoint still wrote to disk"


def _decision():
    from tokenmizer.graph_memory.graph import NodeStatus, NodeType
    return (NodeType.DECISION, "Use PostgreSQL for orders", NodeStatus.COMPLETED)
