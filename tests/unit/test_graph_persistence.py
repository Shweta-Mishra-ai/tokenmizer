"""
Unit tests — graph persistence fixes.

Covers the audit fixes to GraphMemory's storage layer:

1. _dirty flag tracking — _persist() previously rewrote the ENTIRE node
   set as JSON on every call regardless of whether anything changed
   (O(n) write amplification on every chat turn). Now it tracks a dirty
   flag and skips the rewrite when nothing changed since the last
   successful persist.

2. force=True — callers that mutate node/edge state directly without
   going through add_node()/add_edge() (e.g. the /api/decision/invalidate
   endpoint, which flips node.status directly) need an unconditional
   write; force bypasses the dirty check for exactly those cases.

3. decision_tracking_failures / persistence_broken — these used to be
   silent failures visible only in debug-level logs. They're now
   instance-level counters surfaced through stats(), so a session's
   "is my memory actually working" health can be checked programmatically
   instead of by grepping logs.
"""
import os
import time

import pytest

from tokenmizer.graph_memory.graph import GraphMemory
from tokenmizer.graph_memory.types import EdgeType, NodeStatus, NodeType


@pytest.fixture
def graph(tmp_path):
    return GraphMemory("dirty-flag-test-session", storage_dir=str(tmp_path))


class TestDirtyFlagTracking:

    def test_starts_dirty_so_first_persist_always_writes(self, graph):
        assert graph._dirty is True

    def test_add_node_sets_dirty(self, graph):
        graph._persist()
        assert graph._dirty is False
        graph.add_node(NodeType.TASK, "Implement login")
        assert graph._dirty is True

    def test_persist_clears_dirty_on_success(self, graph):
        graph.add_node(NodeType.TASK, "Implement login")
        assert graph._dirty is True
        graph._persist()
        assert graph._dirty is False

    def test_redundant_persist_is_skipped(self, graph):
        graph.add_node(NodeType.TASK, "Implement login")
        graph._persist()
        mtime_before = os.path.getmtime(graph._db_path)
        time.sleep(0.02)
        graph._persist()  # nothing changed — should be a true no-op
        mtime_after = os.path.getmtime(graph._db_path)
        assert mtime_before == mtime_after, "persist() rewrote the DB despite no changes"

    def test_force_bypasses_dirty_check(self, graph):
        graph.add_node(NodeType.TASK, "Implement login")
        graph._persist()
        assert graph._dirty is False
        mtime_before = os.path.getmtime(graph._db_path)
        time.sleep(0.02)
        graph._persist(force=True)
        mtime_after = os.path.getmtime(graph._db_path)
        assert mtime_before != mtime_after, "force=True must write even when not dirty"

    def test_add_edge_sets_dirty(self, graph):
        n1 = graph.add_node(NodeType.TASK, "Implement login")
        n2 = graph.add_node(NodeType.FILE, "src/auth.py")
        graph._persist()
        assert graph._dirty is False
        graph.add_edge(n1, n2, EdgeType.IMPLEMENTS)
        assert graph._dirty is True

    def test_duplicate_edge_does_not_mark_dirty(self, graph):
        n1 = graph.add_node(NodeType.TASK, "Implement login")
        n2 = graph.add_node(NodeType.FILE, "src/auth.py")
        graph.add_edge(n1, n2, EdgeType.IMPLEMENTS)
        graph._persist()
        assert graph._dirty is False
        graph.add_edge(n1, n2, EdgeType.IMPLEMENTS)  # genuine no-op (dedup)
        assert graph._dirty is False, "adding a duplicate edge should not mark dirty"

    def test_dedup_touch_marks_dirty(self, graph):
        """Touching an existing node (dedup path) updates updated_at and
        possibly status — this must still be persisted, not skipped."""
        graph.add_node(NodeType.TASK, "Implement login", status=NodeStatus.PENDING)
        graph._persist()
        assert graph._dirty is False
        graph.add_node(NodeType.TASK, "Implement login", status=NodeStatus.IN_PROGRESS)
        assert graph._dirty is True

    def test_data_survives_reload_with_dirty_optimization_in_place(self, graph, tmp_path):
        """End-to-end: the dirty-flag optimization must never cause data
        loss — a node added and persisted must survive a fresh load."""
        nid = graph.add_node(NodeType.TASK, "Implement login")
        graph._persist()
        reloaded = GraphMemory("dirty-flag-test-session", storage_dir=str(tmp_path))
        assert nid in reloaded._nodes
        assert reloaded._nodes[nid].label == "Implement login"


class TestPersistenceHealthSignals:
    """
    These fields used to be invisible (debug-level log lines only).
    Now surfaced via stats() so a caller can detect degraded persistence
    or broken decision-tracking without grepping server logs.
    """

    def test_decision_tracking_failures_starts_zero(self, graph):
        assert graph._decision_tracking_failures == 0

    def test_persistence_broken_starts_false(self, graph):
        assert graph._persistence_broken is False

    def test_stats_includes_health_fields(self, graph):
        # NOTE: stats() intentionally returns a plain dict (asdict(dto)) —
        # that's the documented serialization boundary for JSON responses,
        # not a bug. Check dict keys, not attribute access.
        graph.add_node(NodeType.TASK, "Implement login feature")
        stats = graph.stats()
        assert "decision_tracking_failures" in stats
        assert "persistence_broken" in stats
        assert stats["decision_tracking_failures"] == 0
        assert stats["persistence_broken"] is False


class TestDirectMutationRequiresForce:
    """
    Regression test for a bug introduced by the dirty-flag fix itself and
    caught only in a later, separate accuracy pass — not in the original
    commit. The dirty-flag optimization (TestDirtyFlagTracking above) is
    correct for the normal add_node()/add_edge() paths, which set _dirty
    themselves. But api/app.py's invalidate_decision() endpoint mutates
    `node.status` DIRECTLY (it has to — there's no add_node-shaped API for
    "change this existing node's status"), bypassing dirty-tracking
    entirely, and then called `graph._persist()` with no `force=True`.

    Result: the dirty flag was still False from the last clean persist,
    so the write was silently skipped. invalidate_decision would return
    HTTP 200 with `"status": "invalidated"` while the actual status
    change never reached disk — the exact silent-data-loss pattern this
    whole audit was supposed to eliminate, reintroduced by one of the
    audit's own earlier fixes. Found by deliberately re-deriving every
    `_persist()` call site after the dirty-flag commit and checking each
    one's mutation pattern, rather than assuming "persistence is already
    fixed" because an earlier commit touched the file.
    """

    def test_persist_without_force_after_direct_mutation_is_lost(self, graph, tmp_path):
        """Demonstrates the BUG pattern in isolation — this is what
        invalidate_decision's old code (graph._persist() with no force)
        would have done. Asserts the data-loss behavior explicitly so a
        future change can't silently reintroduce it without this test
        catching it."""
        nid = graph.add_node(NodeType.DECISION, "Use PostgreSQL for the database")
        graph._persist()
        assert graph._dirty is False

        # Direct mutation — exactly what invalidate_decision does, bypassing add_node
        graph._nodes[nid].status = NodeStatus.INVALIDATED
        graph._persist()  # BUG PATTERN: no force=True — dirty is still False, write skipped

        # Reload from disk: the in-memory mutation must NOT have survived,
        # because the buggy call pattern (no force) never reached the DB.
        # This test exists to make the bug's exact symptom explicit and
        # regression-testable, not to encourage the pattern.
        reloaded = GraphMemory("dirty-flag-test-session", storage_dir=str(tmp_path))
        assert reloaded._nodes[nid].status != NodeStatus.INVALIDATED, (
            "If this assertion fails, _persist()'s dirty-flag behavior has "
            "changed — re-check whether invalidate_decision's force=True "
            "is still necessary, since the premise of the next test "
            "(that force=True is REQUIRED) depends on this skip behavior "
            "actually occurring without it."
        )

    def test_invalidate_decision_pattern_persists_correctly_with_force(self, graph, tmp_path):
        """The ACTUAL fix, verified end-to-end: invalidate_decision's
        direct-mutation-then-persist pattern, with force=True, must
        survive a full reload from disk."""
        nid = graph.add_node(NodeType.DECISION, "Use PostgreSQL for the database")
        graph._persist()
        assert graph._dirty is False

        # Exactly what the fixed invalidate_decision endpoint does:
        graph._nodes[nid].status = NodeStatus.INVALIDATED
        graph._persist(force=True)

        reloaded = GraphMemory("dirty-flag-test-session", storage_dir=str(tmp_path))
        assert reloaded._nodes[nid].status == NodeStatus.INVALIDATED, (
            "invalidate_decision's status change did not survive a reload — "
            "force=True is required after any direct node mutation, or the "
            "dirty-flag optimization silently drops the write"
        )
