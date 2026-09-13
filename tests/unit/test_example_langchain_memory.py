"""examples/langchain_memory.py — TokenMizerChatMessageHistory.

Skipped when langchain-core is not installed (it is not a TokenMizer
dependency — see pyproject.toml's `adapters` extra). No provider/LLM is
exercised here; only the message-history contract and the graph-memory
side channel.
"""
from __future__ import annotations

import pytest

from tests.unit._examples_helpers import load_example

langchain_core = pytest.importorskip("langchain_core")

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

langchain_memory = load_example("langchain_memory.py")


def _history(tmp_path, session_id="lc-test"):
    return langchain_memory.TokenMizerChatMessageHistory(session_id, storage_dir=str(tmp_path))


def test_messages_returns_every_turn_verbatim(tmp_path):
    history = _history(tmp_path)
    history.add_messages([
        HumanMessage("What database should we use?"),
        AIMessage("Decided: use PostgreSQL for order storage."),
    ])
    assert [m.content for m in history.messages] == [
        "What database should we use?",
        "Decided: use PostgreSQL for order storage.",
    ]


def test_clear_empties_the_verbatim_buffer(tmp_path):
    history = _history(tmp_path)
    history.add_messages([HumanMessage("hello")])
    history.clear()
    assert history.messages == []


def test_added_messages_are_extracted_into_graph_memory(tmp_path):
    history = _history(tmp_path)
    history.add_messages([
        HumanMessage("We need to build the checkout service."),
        AIMessage("Decided: use PostgreSQL for order storage."),
    ])
    results = history.search("postgres")
    assert results and "postgres" in results[0]["label"].lower()


def test_context_gives_a_resume_block(tmp_path):
    history = _history(tmp_path)
    history.add_messages([
        HumanMessage("We need to build the checkout service."),
        AIMessage("Decided: use PostgreSQL for order storage."),
    ])
    assert "postgres" in history.context().lower()


def test_non_text_content_is_kept_verbatim_but_not_extracted(tmp_path):
    """A tool message (or any non-string-content message) must not crash
    extraction; it is kept in the verbatim buffer like any other
    message, just not fed to the heuristic extractor."""
    history = _history(tmp_path)
    history.add_messages([
        SystemMessage("You are a helpful assistant."),
        ToolMessage(content="42", tool_call_id="call-1"),
    ])
    assert len(history.messages) == 2


def test_two_sessions_do_not_share_memory(tmp_path):
    a = _history(tmp_path, "lc-a")
    b = _history(tmp_path, "lc-b")
    a.add_messages([
        HumanMessage("We need to build the checkout service."),
        AIMessage("Decided: use PostgreSQL for order storage."),
    ])
    assert b.search("postgres") == []
