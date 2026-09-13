"""examples/langgraph_checkpointer.py — TokenMizerCheckpointSaver.

Skipped when langgraph/langchain-core are not installed (neither is a
TokenMizer dependency — see pyproject.toml's `adapters` extra). The graph
node is a stub, not a real provider call — no network is exercised.
"""
from __future__ import annotations

import pytest

from tests.unit._examples_helpers import load_example

pytest.importorskip("langgraph")
langchain_core = pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402

langgraph_checkpointer = load_example("langgraph_checkpointer.py")
TokenMizerCheckpointSaver = langgraph_checkpointer.TokenMizerCheckpointSaver


def _stub_model(state):
    return {"messages": [AIMessage("Decided: use PostgreSQL for order storage.")]}


def _compiled_app(checkpointer):
    graph = StateGraph(MessagesState)
    graph.add_node("call_model", _stub_model)
    graph.add_edge(START, "call_model")
    graph.add_edge("call_model", END)
    return graph.compile(checkpointer=checkpointer)


def test_checkpoint_durability_still_works(tmp_path):
    """The base InMemorySaver contract — checkpoint replay — must be
    unaffected by the extraction side channel."""
    checkpointer = TokenMizerCheckpointSaver(storage_dir=str(tmp_path))
    app = _compiled_app(checkpointer)
    config = {"configurable": {"thread_id": "lg-durability"}}

    app.invoke({"messages": [HumanMessage("hello")]}, config=config)
    state = app.get_state(config)

    assert any(isinstance(m, AIMessage) for m in state.values["messages"])


def test_graph_run_is_extracted_into_memory(tmp_path):
    checkpointer = TokenMizerCheckpointSaver(storage_dir=str(tmp_path))
    app = _compiled_app(checkpointer)
    config = {"configurable": {"thread_id": "lg-extract"}}

    app.invoke(
        {"messages": [HumanMessage("We need to build the checkout service.")]},
        config=config,
    )

    results = checkpointer.memory_for("lg-extract").search("postgres")
    assert results and "postgres" in results[0]["label"].lower()


def test_two_threads_do_not_share_memory(tmp_path):
    checkpointer = TokenMizerCheckpointSaver(storage_dir=str(tmp_path))
    app = _compiled_app(checkpointer)

    app.invoke(
        {"messages": [HumanMessage("We need to build the checkout service.")]},
        config={"configurable": {"thread_id": "lg-a"}},
    )

    assert checkpointer.memory_for("lg-b").search("postgres") == []


def test_extraction_failure_does_not_break_the_checkpoint(tmp_path, monkeypatch):
    """The checkpoint write itself must succeed even if the bonus
    graph-memory extraction raises."""
    checkpointer = TokenMizerCheckpointSaver(storage_dir=str(tmp_path))

    def broken_memory_for(thread_id):
        raise RuntimeError("simulated extraction failure")

    monkeypatch.setattr(checkpointer, "memory_for", broken_memory_for)
    app = _compiled_app(checkpointer)
    config = {"configurable": {"thread_id": "lg-broken"}}

    app.invoke({"messages": [HumanMessage("hello")]}, config=config)
    state = app.get_state(config)

    assert any(isinstance(m, AIMessage) for m in state.values["messages"])
