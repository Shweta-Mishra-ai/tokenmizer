"""
Regression tests for TM-04's second half: checkpoint retention.

Background: with the context_pct accumulator bug fixed (see
test_context_tracking.py), the auto-checkpoint trigger no longer fires on
every turn once crossed — but a long real session that genuinely stays
near the trigger threshold can still legitimately fire many auto
checkpoints over its lifetime. Each one snapshots the FULL graph as JSON
(see issue #27) with no retention policy anywhere, so the `checkpoints`
table grows without bound for a long-lived session.

Fix: cap the number of AUTO-triggered checkpoints retained per session
(oldest pruned first). Manual checkpoints (an explicit user action) are
never pruned by this policy — only ones the trigger itself created.
"""
from __future__ import annotations

import pytest

from tokenmizer.checkpoints.manager import CheckpointManager
from tokenmizer.graph_memory.graph import GraphMemory

MESSAGES = [
    {"role": "user", "content": "Use PostgreSQL for the primary datastore"},
    {"role": "assistant", "content": "Decided: use PostgreSQL for the primary datastore."},
]


@pytest.fixture
def graph(tmp_path):
    g = GraphMemory("retention-test", storage_dir=str(tmp_path))
    g.extract_from_messages(MESSAGES, incremental=False)
    return g


class TestAutoCheckpointRetention:

    def test_auto_checkpoints_beyond_cap_are_pruned(self, tmp_path, graph):
        mgr = CheckpointManager(storage_dir=str(tmp_path), auto_retention=5)
        for i in range(12):
            mgr.create(
                session_id="retention-test", messages=MESSAGES, graph=graph,
                context_pct=0.85, trigger="auto_threshold",
            )
        checkpoints = mgr.list_checkpoints("retention-test")
        assert len(checkpoints) == 5, (
            f"expected auto-checkpoint retention to cap the table at 5 "
            f"rows for this session, got {len(checkpoints)}"
        )

    def test_retention_keeps_the_most_recent(self, tmp_path, graph):
        mgr = CheckpointManager(storage_dir=str(tmp_path), auto_retention=3)
        ids = []
        for i in range(6):
            ckpt = mgr.create(
                session_id="retention-order-test", messages=MESSAGES, graph=graph,
                context_pct=0.85, trigger="auto_threshold",
            )
            ids.append(ckpt.checkpoint_id)
        remaining = {c["checkpoint_id"] for c in mgr.list_checkpoints("retention-order-test")}
        assert remaining == set(ids[-3:]), (
            "retention must keep the NEWEST checkpoints, not an arbitrary subset"
        )

    def test_manual_checkpoints_are_never_pruned(self, tmp_path, graph):
        """A manual checkpoint is an explicit user action (or the
        /api/checkpoint endpoint / CLI). The auto-retention cap must never
        silently delete one, even if many auto checkpoints follow it."""
        mgr = CheckpointManager(storage_dir=str(tmp_path), auto_retention=3)
        manual = mgr.create(
            session_id="manual-preserved-test", messages=MESSAGES, graph=graph,
            context_pct=0.0, trigger="manual",
        )
        for i in range(10):
            mgr.create(
                session_id="manual-preserved-test", messages=MESSAGES, graph=graph,
                context_pct=0.85, trigger="auto_threshold",
            )
        remaining = {c["checkpoint_id"] for c in mgr.list_checkpoints("manual-preserved-test")}
        assert manual.checkpoint_id in remaining, (
            "auto-checkpoint retention pruned a MANUAL checkpoint — manual "
            "checkpoints must never be pruned by the automatic-trigger cap"
        )

    def test_retention_is_scoped_per_session(self, tmp_path, graph):
        """Session A's auto checkpoints must not count against session B's
        retention cap, and vice versa."""
        g2 = GraphMemory("retention-session-b", storage_dir=str(tmp_path))
        g2.extract_from_messages(MESSAGES, incremental=False)

        mgr = CheckpointManager(storage_dir=str(tmp_path), auto_retention=3)
        for _ in range(6):
            mgr.create(session_id="retention-session-a", messages=MESSAGES,
                       graph=graph, context_pct=0.85, trigger="auto_threshold")
        for _ in range(2):
            mgr.create(session_id="retention-session-b", messages=MESSAGES,
                       graph=g2, context_pct=0.85, trigger="auto_threshold")

        assert len(mgr.list_checkpoints("retention-session-a")) == 3
        assert len(mgr.list_checkpoints("retention-session-b")) == 2

    def test_default_retention_is_reasonable(self, tmp_path, graph):
        """No explicit auto_retention passed — the default must still cap
        growth for a long session rather than growing unboundedly."""
        mgr = CheckpointManager(storage_dir=str(tmp_path))
        for _ in range(60):
            mgr.create(session_id="default-retention-test", messages=MESSAGES,
                       graph=graph, context_pct=0.85, trigger="auto_threshold")
        assert len(mgr.list_checkpoints("default-retention-test")) < 60, (
            "checkpoints table grew without bound under default settings"
        )
