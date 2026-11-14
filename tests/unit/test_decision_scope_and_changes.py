"""Which decision replaces which.

Two failures found by benchmarks/soak and a scenario sweep, present on main:
a decision's scope was not part of its identity (the ledger service's Kafka
topic "superseded" the billing service's), and a change of mind that names
the slot rather than the old value ("switch the partition key to X", "store
sessions in Y instead") extracted nothing, so the old choice stayed current.
"""
from __future__ import annotations

import pytest

from tokenmizer.graph_memory.decision_tracker import decision_scope
from tokenmizer.graph_memory.graph import GraphMemory, NodeType
from tokenmizer.graph_memory.patterns import find_slot_changes, find_supersessions


def _decisions(tmp_path, *turns):
    g = GraphMemory("d", storage_dir=str(tmp_path))
    history: list[dict] = []
    for t in turns:
        history += [{"role": "user", "content": "next?"}, {"role": "assistant", "content": t}]
        g.extract_from_messages(history)
    return {n.label: n.status.value for n in g._nodes.values() if n.type == NodeType.DECISION}


def _status_of(decisions, fragment):
    found = [s for label, s in decisions.items() if fragment in label]
    assert found, (fragment, decisions)
    return found[0]


# ── a change of mind supersedes what it changes ─────────────────────────────

@pytest.mark.parametrize("first,second,old,new", [
    ("We'll use `tenant_id` as the partition key.",
     "Change of plan: switch the partition key to `account_id`.", "tenant_id", "account_id"),
    ("Decision: store sessions in `redis_cache`.",
     "Actually, let's store sessions in `memcached_pool` instead.", "redis_cache", "memcached_pool"),
    ("Decision: `gpt-4o-mini` as the default model.",
     "Let's change the default model to claude-sonnet-5.", "gpt-4o-mini", "claude-sonnet-5"),
    ("We use Jest for tests.", "Let's replace Jest with Vitest.", "Jest", "Vitest"),
    ("We'll use Rust for performance.",
     "Decision: switching from Rust to Go for simplicity.", "Rust for performance", "Go"),
])
def test_a_change_of_mind_supersedes_the_old_choice(tmp_path, first, second, old, new):
    d = _decisions(tmp_path, first, second)
    assert _status_of(d, old) == "superseded"
    assert _status_of(d, new) == "completed"


def test_a_question_about_switching_is_not_a_switch():
    from tokenmizer.graph_memory.hybrid_extractor import get_hybrid_extractor
    r = get_hybrid_extractor("coding").heuristic_extract(
        [{"role": "user", "content": "Should we switch the partition key to `account_id`?"}])
    assert not r.decisions and not r.superseded


@pytest.mark.parametrize("text", ["switch the button to the left side",
                                  "We will use Postgres instead of MySQL."])
def test_not_a_slot_change(text):
    assert find_slot_changes(text) == []


def test_the_imperative_replace_is_read():
    assert [(o, n) for o, n, *_ in find_supersessions("Let's replace Jest with Vitest.")] \
        == [("Jest", "Vitest")]


# ── different scopes are different decisions ────────────────────────────────

@pytest.mark.parametrize("first,second,a,b", [
    ("Decision: the billing service uses `billing_events` as its Kafka topic.",
     "Decision: the ledger service uses `ledger_events` as its Kafka topic.",
     "billing_events", "ledger_events"),
    ("Decision: for the amber billing we'll use `amber_key` as the partition key.",
     "Decision: for the marble ledger we'll use `marble_key` as the partition key.",
     "amber_key", "marble_key"),
    ("Use Redis for caching.", "Use Redis for rate limiting.", "caching", "rate limiting"),
    ("For the billing service we'll use `gpt-4o-mini` as the default model.",
     "For the search service, change the default model to claude-sonnet-5.",
     "gpt-4o-mini", "claude-sonnet-5"),
])
def test_decisions_for_different_scopes_both_stand(tmp_path, first, second, a, b):
    d = _decisions(tmp_path, first, second)
    assert _status_of(d, a) == "completed"
    assert _status_of(d, b) == "completed"


def test_the_same_scope_still_supersedes(tmp_path):
    d = _decisions(tmp_path,
                   "For the billing service we'll use `gpt-4o-mini` as the default model.",
                   "For the billing service, change the default model to claude-sonnet-5.")
    assert _status_of(d, "gpt-4o-mini") == "superseded"
    assert _status_of(d, "claude-sonnet-5") == "completed"


@pytest.mark.parametrize("label,scope", [
    ("the billing service uses billing_events", {"billing"}),
    ("Use Redis for caching", {"caching"}),
    ("Use Postgres for the orders table because it locks rows", {"orders", "table"}),
    ("Use Rust for performance", set()),       # a reason, not what it covers
    ("Use Redis for sessions", set()),
])
def test_decision_scope(label, scope):
    assert decision_scope(label) == scope


def test_a_fronted_scope_stays_in_the_label(tmp_path):
    d = _decisions(tmp_path, "For the search service we'll use Elasticsearch.")
    assert any("search service" in label for label in d), d
