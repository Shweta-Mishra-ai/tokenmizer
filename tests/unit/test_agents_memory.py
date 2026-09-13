"""tokenmizer.agents.Memory — the in-process agent interface."""
from __future__ import annotations

import pytest

from tokenmizer.agents import Memory

SESSION = [
    {"role": "user", "content": "We are building the checkout service and need a datastore."},
    {"role": "assistant", "content":
     "Decided: PostgreSQL for order storage. Created internal/store/postgres.go."},
    {"role": "user", "content": "The integration test hangs forever."},
    {"role": "assistant", "content":
     "Found it: the gRPC client had no dial timeout. Fixed by adding a 5 second "
     "context timeout in internal/order/client.go."},
    {"role": "user", "content": "Actually drop that library, write the retry by hand."},
    {"role": "assistant", "content":
     "Switching from cenkalti/backoff to a hand-rolled retry loop."},
]


@pytest.fixture
def memory(tmp_path):
    return Memory("agent-test", storage_dir=str(tmp_path))


def test_add_returns_how_many_nodes_were_gained(memory):
    gained = memory.add(SESSION)
    assert gained > 0
    assert memory.add(SESSION) == 0, "re-adding the same messages must be a no-op"


def test_search_returns_plain_dicts_not_node_objects(memory):
    memory.add(SESSION)
    results = memory.search("what are we storing orders in")
    assert results, "nothing found"
    assert all(isinstance(r, dict) for r in results)
    assert {"type", "label", "summary", "status", "confidence"} <= set(results[0])
    assert any("postgres" in r["label"].lower() for r in results)


def test_search_can_filter_by_type(memory):
    memory.add(SESSION)
    only_files = memory.search("client", types=["file"])
    assert only_files and all(r["type"] == "file" for r in only_files)


def test_decisions_and_errors_views(memory):
    memory.add(SESSION)
    assert any("postgres" in d["label"].lower() for d in memory.decisions())
    assert any("hang" in e["label"].lower() or "timeout" in e["label"].lower()
               for e in memory.errors())


def test_context_is_a_resume_block(memory):
    memory.add(SESSION)
    block = memory.context(token_budget=400)
    assert "postgres" in block.lower()


def test_why_traces_a_supersession(memory):
    memory.add(SESSION)
    trail = memory.why("retry")
    assert isinstance(trail, dict)


def test_memory_survives_a_new_instance_on_the_same_store(tmp_path):
    first = Memory("persist", storage_dir=str(tmp_path))
    first.add(SESSION)
    first.save()

    second = Memory("persist", storage_dir=str(tmp_path))
    assert second.search("orders"), "nothing came back from disk"


def test_sessions_are_isolated(tmp_path):
    a = Memory("a", storage_dir=str(tmp_path))
    b = Memory("b", storage_dir=str(tmp_path))
    a.add(SESSION)
    assert b.search("orders") == []


def test_default_session_id_comes_from_the_working_directory(tmp_path, monkeypatch):
    project = tmp_path / "checkout-service"
    project.mkdir()
    monkeypatch.chdir(project)
    assert Memory(storage_dir=str(tmp_path)).session_id == "checkout-service"


def test_no_derivable_session_id_is_an_explicit_error(tmp_path, monkeypatch):
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.chdir(scratch)
    with pytest.raises(ValueError, match="session_id"):
        Memory(storage_dir=str(tmp_path))


def test_search_results_are_tagged_with_their_own_session_id(memory):
    memory.add(SESSION)
    results = memory.search("postgres")
    assert results
    assert all(r["session_id"] == "agent-test" for r in results)


def test_cross_session_recall_is_off_by_default(tmp_path):
    a = Memory("cs-a", storage_dir=str(tmp_path))
    b = Memory("cs-b", storage_dir=str(tmp_path))
    a.add(SESSION)
    assert b.search("orders") == []


def test_cross_session_recall_finds_other_sessions_when_enabled(tmp_path):
    a = Memory("cs-a2", storage_dir=str(tmp_path), cross_session_recall=True)
    b = Memory("cs-b2", storage_dir=str(tmp_path), cross_session_recall=True)
    a.add(SESSION)

    results = b.search("what are we storing orders in")

    assert results, "expected session cs-a2's node to be found from cs-b2"
    assert any(r["session_id"] == "cs-a2" for r in results)
    assert any("postgres" in r["label"].lower() for r in results)


def test_cross_session_recall_defaults_from_settings(tmp_path, monkeypatch):
    import tokenmizer.config.settings as settings_module

    monkeypatch.setattr(
        settings_module.get_settings().graph_checkpoint, "cross_session_recall", True
    )
    a = Memory("cs-a3", storage_dir=str(tmp_path))
    b = Memory("cs-b3", storage_dir=str(tmp_path))
    a.add(SESSION)

    assert b.search("orders"), "Memory() should follow the settings default when not overridden"


def test_works_with_no_server_no_key_no_network(memory, monkeypatch):
    """The point of the module. If anything here tries the network it is a
    regression."""
    import socket

    def refuse(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.delenv("TOKENMIZER_ANTHROPIC_API_KEY", raising=False)
    memory.add(SESSION)
    assert memory.search("orders")
