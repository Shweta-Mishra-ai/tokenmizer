"""
Regression tests for TM-07: importance decay must be a function of
elapsed WALL-CLOCK TIME, not of how many times apply_importance_decay()
happens to be called.

Bug: apply_importance_decay() computed `decay_factor` from the node's
absolute age (age_days(), based on updated_at) and multiplied it into the
CURRENT importance — but nothing recorded when decay was last applied.
Since apply_importance_decay() runs once per chat turn (inside
extract_from_messages()), and a node's age doesn't change meaningfully
between turns seconds apart, the SAME decay factor got reapplied to an
already-decayed value on every single turn. Reproduced empirically: a
10-day-old COMPLETED task at importance=0.8 collapsed to the 0.10 floor
within 3 calls made at the same instant — three chatty turns, no time
passing at all.

Fix: MemoryNode now tracks last_decayed_at, and decay magnitude is
computed from elapsed time SINCE THE LAST DECAY APPLICATION, not from
absolute node age. Calling apply_importance_decay() twice in immediate
succession should decay a node by approximately nothing the second time
(no time passed) instead of compounding the first call's reduction.
"""
from __future__ import annotations

import time

import pytest

from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType


@pytest.fixture
def graph(tmp_path):
    return GraphMemory("decay-idempotence-test", storage_dir=str(tmp_path))


class TestDecayDoesNotCompoundWithinTheSameInstant:

    def test_repeated_calls_at_same_instant_do_not_collapse_importance(self, graph):
        nid = graph.add_node(NodeType.TASK, "Implement the user authentication flow",
                             NodeStatus.COMPLETED, importance=0.8)
        node = graph._nodes[nid]
        # 10 days old, well past the 3-day grace period for COMPLETED tasks
        node.created_at = time.time() - 10 * 86400
        node.updated_at = node.created_at
        node.last_decayed_at = node.created_at

        graph.apply_importance_decay()
        after_first = node.importance
        assert after_first < 0.8, "decay should have applied at all on the first call"

        for _ in range(5):
            graph.apply_importance_decay()

        assert node.importance == pytest.approx(after_first, abs=0.01), (
            f"repeated decay calls at the SAME instant kept reducing "
            f"importance ({after_first} -> {node.importance}) — decay is "
            f"compounding per CALL instead of per unit of elapsed time"
        )
        assert node.importance > 0.10, (
            "importance collapsed to the floor from repeated same-instant "
            "calls — this is exactly the bug being regression-tested"
        )

    def test_decay_does_progress_across_real_elapsed_time(self, graph):
        """Confirms the fix doesn't accidentally disable decay entirely —
        simulating real elapsed time between calls must still decay."""
        nid = graph.add_node(NodeType.TASK, "Implement the user authentication flow",
                             NodeStatus.COMPLETED, importance=0.8)
        node = graph._nodes[nid]
        node.created_at = time.time() - 10 * 86400
        node.updated_at = node.created_at
        node.last_decayed_at = node.created_at

        graph.apply_importance_decay()
        after_first = node.importance

        # Simulate 5 more real days passing before the next decay call.
        node.last_decayed_at -= 5 * 86400
        graph.apply_importance_decay()

        assert node.importance < after_first, (
            "importance should continue decaying once real time actually "
            "passes since the last decay application"
        )

    def test_many_turns_in_a_short_real_session_barely_decay(self, graph):
        """The realistic scenario this bug actually broke: a chatty
        session firing apply_importance_decay() every turn over a few
        real minutes must not visibly erode importance just from turn
        count. The node is old (out of its grace period), but decay
        bookkeeping is already caught up to "now" — as it would be in a
        long-lived process that's been calling apply_importance_decay()
        regularly all along — so this specifically isolates the
        per-call-not-per-time-unit compounding bug, separate from the
        legitimate one-time catch-up decay a stale last_decayed_at would
        cause (covered by the previous test)."""
        nid = graph.add_node(NodeType.TASK, "Implement the user authentication flow",
                             NodeStatus.COMPLETED, importance=0.8)
        node = graph._nodes[nid]
        node.created_at = time.time() - 10 * 86400
        node.updated_at = node.created_at
        node.last_decayed_at = time.time()  # decay bookkeeping already caught up

        for _ in range(20):  # 20 turns, no real time passing between them
            graph.apply_importance_decay()

        assert node.importance == pytest.approx(0.8, abs=0.01), (
            f"20 same-instant turns eroded importance to {node.importance} "
            f"— decay must scale with elapsed time, not turn count"
        )


class TestDecayStillRespectsExistingSafetyRules:
    """Guard against the fix accidentally loosening the rules the
    original decay design already got right."""

    def test_floor_is_still_010(self, graph):
        nid = graph.add_node(NodeType.DECISION, "Use MongoDB for the event log",
                             NodeStatus.SUPERSEDED, importance=0.2)
        node = graph._nodes[nid]
        node.valid_until = time.time() - 90 * 86400
        node.created_at = time.time() - 100 * 86400
        node.updated_at = node.created_at
        node.last_decayed_at = node.created_at - 365 * 86400  # a huge elapsed gap

        for _ in range(10):
            graph.apply_importance_decay()
        assert node.importance >= 0.10

    def test_active_decisions_never_decay(self, graph):
        nid = graph.add_node(NodeType.DECISION, "Use PostgreSQL for the primary datastore",
                             NodeStatus.COMPLETED, importance=0.9)
        node = graph._nodes[nid]
        node.created_at = time.time() - 365 * 86400
        node.updated_at = node.created_at
        node.last_decayed_at = node.created_at
        graph.apply_importance_decay()
        assert node.importance == 0.9

    def test_goals_never_decay(self, graph):
        nid = graph.add_node(NodeType.GOAL, "Build a production auth service", importance=1.0)
        node = graph._nodes[nid]
        node.created_at = time.time() - 365 * 86400
        node.updated_at = node.created_at
        node.last_decayed_at = node.created_at
        graph.apply_importance_decay()
        assert node.importance == 1.0
