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


# ── a test run's traceback outranks its list of failing test names ──────────

def test_the_exception_survives_a_long_list_of_failing_tests(tmp_path):
    """A test run names every failing test before the traceback that says
    what broke. The per-message limit kept the first labels it saw — the
    test names — and dropped the exception (found on a real SWE-bench
    session, django-14608)."""
    names = "\n".join(f"ERROR: test_case_{i:03d} (tests.forms.test_formsets.Case{i:03d})"
                      for i in range(80))
    trace = ("Traceback (most recent call last):\n"
             '  File "tests/forms/test_formsets.py", line 1408, in test_valid\n'
             "    formset = self.make_choiceformset([])\n"
             "AttributeError: 'AllValidTests' object has no attribute 'make_choiceformset'")
    g = _graph(tmp_path)
    # SWE-agent returns command output as a user turn, not a tool result, so
    # only the text patterns read it.
    g.extract_from_messages([{"role": "assistant", "content": "Running the formset tests."},
                             {"role": "user", "content": names + "\n\n" + trace}])
    assert any("make_choiceformset" in label for label in _labels(g, NodeType.ERROR))


def test_being_listed_again_does_not_make_a_file_important(tmp_path):
    """A re-mention raises a node's importance; a path re-listed by `ls`
    a dozen times climbed from 0.35 to 1.0 that way and, in a 1,000-turn
    soak, pushed every file the agent named out of the graph."""
    g = _graph(tmp_path)
    history = [{"role": "user", "content": "go"}]
    for i in range(15):
        history += [{"role": "user", "content": f"list {i}"}, *_tool(i, _listing(60))]
        g.extract_from_messages(history)
    listed = [n for n in g._nodes.values() if n.type == NodeType.FILE]
    assert listed and max(n.importance for n in listed) < 0.5


# ── a question in words finds a file named in code ──────────────────────────

def test_a_file_is_found_by_the_words_in_its_name(tmp_path):
    """People ask "the quartz uploader"; code writes quartz_uploader. Matched
    on whole tokens only, and with "for"/"the" counting as overlap, the
    decisions below outranked the file in every one of ten probes of a
    1,000-turn session."""
    g = _graph(tmp_path)
    for d in ("Redis for refresh token storage", "PostgreSQL for order storage",
              "for the nimbus scheduler we'll use nimbus_key as the partition key",
              "CSS variables for theming"):
        g.add_node(NodeType.DECISION, d, NodeStatus.COMPLETED, importance=0.95, confidence=0.9)
    f = g.add_node(NodeType.FILE, "src/quartz_uploader/quartz_uploader_s03.py",
                   NodeStatus.IN_PROGRESS, importance=0.7)
    top = [n.id for n in g.query("Which file did we change for the quartz uploader?", top_k=3)]
    assert f in top


def test_directory_scaffolding_is_not_a_match(tmp_path):
    from tokenmizer.graph_memory.graph import GraphMemory as G
    words = G._query_tokens("tests/conftest.py src/app/routes.py")
    assert {"conftest", "routes"} <= words
    assert not {"test", "tests", "src", "app", "py"} & words
