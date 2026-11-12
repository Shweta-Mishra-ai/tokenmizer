"""What a long, large session keeps.

Each test is a data-loss path found by benchmarks/soak: before the fix, a
1,000-turn session kept 23 of 50 planted facts in its graph.
"""
from __future__ import annotations

import random

import pytest

from tokenmizer.graph_memory.decision_tracker import (
    _identifiers,
    _is_same_decision,
    _is_same_error,
)
from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType


def _graph(tmp_path, name="s", **kw):
    return GraphMemory(name, storage_dir=str(tmp_path / name), **kw)


def _tool(i, text):
    return [{"role": "assistant", "content": "", "tool_calls": [{
                "id": f"c{i}", "type": "function",
                "function": {"name": "bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": text}]


def _labels(g, node_type):
    return [n.label for n in g._nodes.values() if n.type == node_type and not n._evicted]


# ── identifiers say WHICH thing ────────────────────────────────────────────

def test_the_same_error_in_two_services_is_two_errors(tmp_path):
    g = _graph(tmp_path)
    msgs = [{"role": "user", "content": "go"}]
    for i, svc in enumerate(("amber_billing", "marble_ledger")):
        msgs += _tool(i, f'Traceback (most recent call last):\n  File "src/{svc}/handler.py", '
                         f'line 42, in run\nRuntimeError: {svc}_flush overflowed the batch buffer')
        msgs.append({"role": "user", "content": "next"})
    g.extract_from_messages(msgs, incremental=False)
    errors = " ".join(_labels(g, NodeType.ERROR))
    assert "amber_billing_flush" in errors and "marble_ledger_flush" in errors


def test_the_same_decision_for_two_components_is_two_decisions(tmp_path):
    g = _graph(tmp_path)
    g.extract_from_messages([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "Decision: for the billing service we'll use "
                                         "`billing_tenant_key` as the partition key."},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "Decision: for the ledger service we'll use "
                                         "`ledger_tenant_key` as the partition key."},
    ], incremental=False)
    text = " ".join(n.label + " " + n.summary for n in g._nodes.values()
                    if n.type == NodeType.DECISION)
    assert "billing_tenant_key" in text and "ledger_tenant_key" in text


@pytest.mark.parametrize("a,b", [
    # A recurrence after an edit: only the line number moved.
    ("RuntimeError: flush_batch overflowed at src/a/handler.py line 42",
     "RuntimeError: flush_batch overflowed at src/a/handler.py line 57"),
    # One report names the file, the other doesn't.
    ("KeyError: 'user_id'", "KeyError: 'user_id' in src/api/auth.py"),
])
def test_one_failure_reported_twice_still_merges(a, b):
    assert _is_same_error(a, b)


def test_different_keys_are_different_errors():
    assert not _is_same_error("KeyError: 'user_id'", "KeyError: 'order_id'")


@pytest.mark.parametrize("a,b", [("Use Next.js", "Use NextJS"),
                                 ("Use TypeScript", "use typescript"),
                                 ("Use Redis for sessions",
                                  "use Redis for sessions (see config.py)")])
def test_spellings_of_one_technology_are_not_identifiers(a, b):
    assert _is_same_decision(a, b)


def test_identifier_shapes():
    got = _identifiers("TypeScript getUserName user_id src/x.py config.yaml Next.js "
                       "RuntimeError 0x7f3a9b2c /tmp/pytest-3/a.py macOS")
    assert got == {"getusername", "user_id", "src/x.py", "config.yaml"}


# ── one call over a long history keeps everything ─────────────────────────

def _todo_session(n=30):
    msgs = []
    for i in range(n):
        msgs.append({"role": "user", "content": f"Turn {i}: TODO: write the svc{i:02d}x exporter docs."})
        msgs.append({"role": "assistant", "content":
                     f"Noted. I also fixed the bug in src/svc{i:02d}x/service.py where "
                     f"the svc{i:02d}x cache was never invalidated."})
    return msgs


def _counts(g):
    pending = sum(1 for n in g._nodes.values()
                  if n.type == NodeType.TASK and n.status == NodeStatus.PENDING)
    done = sum(1 for n in g._nodes.values()
               if n.type == NodeType.TASK and n.status == NodeStatus.COMPLETED)
    return pending, done


def test_a_cold_start_on_a_long_history_keeps_every_task(tmp_path):
    """The proxy's first request on an existing conversation carries the
    whole history in one call. A 20-message window and a cap of 15 per
    category kept 1 of 30 TODOs and 1 of 30 fixes."""
    msgs = _todo_session()
    cold = _graph(tmp_path, "cold")
    cold.extract_from_messages(msgs)
    one_by_one = _graph(tmp_path, "inc")
    for k in range(1, len(msgs) + 1):
        one_by_one.extract_from_messages(msgs[:k])
    assert _counts(cold) == _counts(one_by_one) == (30, 30)


# ── a listing does not push real work out ──────────────────────────────────

def _listing(n_lines, seed=1):
    rng = random.Random(seed)
    return "\n".join(f"{rng.randint(100, 99999):>6} src/pkg_{i % 37}/mod_{i}.py"
                     for i in range(n_lines))


def test_files_from_a_bulk_listing_rank_below_files_named_on_their_own(tmp_path):
    g = _graph(tmp_path)
    g.extract_from_messages([{"role": "user", "content": "count lines"},
                             *_tool(1, _listing(60))], incremental=False)
    listed = [n for n in g._nodes.values() if n.type == NodeType.FILE]
    assert len(listed) >= 50                        # still recorded
    assert all(n.importance < 0.5 for n in listed)  # but as listed, not worked on
    g.extract_from_messages([{"role": "assistant",
                              "content": "I edited src/pkg_3/mod_3.py to add the guard."}],
                            incremental=False)
    named = next(n for n in g._nodes.values() if n.label.endswith("pkg_3/mod_3.py"))
    assert named.importance >= 0.7


def test_open_todos_survive_a_flood_of_listed_files(tmp_path):
    g = _graph(tmp_path, max_nodes=100)
    history = [{"role": "user", "content": "TODO: add the retry_budget_x7 to the uploader."},
               {"role": "assistant", "content": "Noted."}]
    for i in range(6):
        history += [{"role": "user", "content": f"list batch {i}"}, *_tool(i, _listing(60, i))]
        g.extract_from_messages(history)
    todos = _labels(g, NodeType.TASK)
    assert any("retry_budget_x7" in t for t in todos), todos
