"""examples/agent_loop.py — the plain (no framework) Memory usage pattern."""
from __future__ import annotations

from tests.unit._examples_helpers import load_example
from tokenmizer.agents import Memory

agent_loop = load_example("agent_loop.py")


def test_run_turn_records_the_exchange_into_memory(tmp_path):
    memory = Memory("loop-test", storage_dir=str(tmp_path))
    calls = []

    def stub_model(system_prompt, user_message):
        calls.append((system_prompt, user_message))
        return "Decided: use PostgreSQL for order storage."

    reply = agent_loop.run_turn(
        memory, "What database should we use?", stub_model,
    )

    assert reply == "Decided: use PostgreSQL for order storage."
    assert len(calls) == 1
    assert memory.search("postgres"), "the reply should have been extracted into memory"


def test_second_turn_sees_the_first_turns_context(tmp_path):
    memory = Memory("loop-test-2", storage_dir=str(tmp_path))
    seen_prompts = []

    def stub_model(system_prompt, user_message):
        seen_prompts.append(system_prompt)
        return "Decided: use PostgreSQL for order storage."

    agent_loop.run_turn(memory, "What database should we use?", stub_model)
    agent_loop.run_turn(memory, "Why did we choose that?", stub_model)

    assert "postgres" not in seen_prompts[0].lower(), \
        "nothing to recall yet on the first turn"
    assert "postgres" in seen_prompts[1].lower(), \
        "the second turn's prompt should carry the first turn's decision"


def test_main_runs_end_to_end_with_no_network(monkeypatch):
    """The point of this example. If anything here tries the network it
    is a regression."""
    import socket

    def refuse(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", refuse)
    agent_loop.main()
