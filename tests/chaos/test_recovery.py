"""
Chaos tests — verify graceful recovery from failures.
"""

import sqlite3

from tokenmizer.checkpoints.manager import CheckpointManager
from tokenmizer.graph_memory.graph import GraphMemory, NodeType


class TestCorruptedGraph:
    def test_corrupted_db_handled_gracefully(self, tmp_path):
        """If the SQLite DB is corrupted, GraphMemory should start fresh."""
        db_path = tmp_path / "graph_memory.db"
        db_path.write_bytes(b"this is not valid sqlite data at all!!")

        # Should not raise — should log warning and start fresh
        g = GraphMemory("chaos-session", storage_dir=str(tmp_path))
        assert len(g._nodes) == 0  # starts fresh

    def test_partial_write_recovery(self, tmp_path):
        """Add nodes, corrupt the DB mid-write, new instance should still load what's good."""
        g = GraphMemory("partial-session", storage_dir=str(tmp_path))
        g.add_node(NodeType.TASK, "Task before corruption")
        g._persist()

        # Now corrupt the processed_hashes (simulate partial write)
        db_path = tmp_path / "graph_memory.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE graphs SET processed_hashes = ? WHERE session_id = ?",
                ("not valid json at all", "partial-session"),
            )
            conn.commit()

        # Should load what it can
        g2 = GraphMemory("partial-session", storage_dir=str(tmp_path))
        # FIXED: this used to assert `len(g2._nodes) >= 0`, which is always
        # true (length can never be negative) — the test ran and "passed"
        # while masking a real bug: corrupted processed_hashes caused the
        # ENTIRE _load() to abort before the node-population loop ran, so
        # a perfectly valid nodes_json got silently discarded too. The
        # node added before corruption MUST survive — that's the whole
        # point of "partial write recovery".
        assert len(g2._nodes) == 1, (
            f"Node written before hash corruption was lost on reload — "
            f"got {len(g2._nodes)} nodes, expected 1. This means "
            f"GraphMemory._load() is letting an unrelated processed_hashes "
            f"parse failure wipe out valid nodes_json data."
        )
        labels = [n.label for n in g2._nodes.values()]
        assert "Task before corruption" in labels, f"Wrong node survived: {labels}"

    def test_add_node_after_load_failure(self, tmp_path):
        """Even if load fails, we should be able to add nodes."""
        db_path = tmp_path / "graph_memory.db"
        db_path.write_bytes(b"corrupted!")

        g = GraphMemory("fresh-after-corrupt", storage_dir=str(tmp_path))
        nid = g.add_node(NodeType.TASK, "Implement fresh task after recovery")
        assert nid in g._nodes


class TestCheckpointChaos:
    def test_checkpoint_with_empty_graph(self, tmp_path):
        """Checkpointing a session with no data should not crash."""
        mgr = CheckpointManager(storage_dir=str(tmp_path))
        g = GraphMemory("empty-session", storage_dir=str(tmp_path))
        # Empty graph — should still create checkpoint without error
        ckpt = mgr.create(
            session_id="empty-session",
            messages=[],
            graph=g,
            context_pct=0.0,
        )
        assert ckpt is not None
        # FIXED: this was `assert ckpt.resume_standard == "" or len(...) >= 0`,
        # which is mathematically always true (len() can't be negative) — same
        # vacuous pattern as the bug fixed above in this file. resume_standard
        # must at minimum be a string (not None, not an exception object) for
        # an empty-graph checkpoint.
        assert isinstance(ckpt.resume_standard, str), (
            f"resume_standard should be a string even for an empty session, "
            f"got {type(ckpt.resume_standard)}"
        )

    def test_checkpoint_with_no_messages(self, tmp_path):
        """Checkpoint with messages=[] should not crash."""
        mgr = CheckpointManager(storage_dir=str(tmp_path))
        g = GraphMemory("no-msgs", storage_dir=str(tmp_path))
        g.add_node(NodeType.TASK, "Some task")
        ckpt = mgr.create(
            session_id="no-msgs",
            messages=[],
            graph=g,
            context_pct=0.5,
        )
        assert ckpt.message_count == 0

    def test_corrupted_checkpoint_db(self, tmp_path):
        """If checkpoint DB is corrupted, get_latest should return None (not crash)."""
        db_path = tmp_path / "checkpoints.db"
        db_path.write_bytes(b"not sqlite data")

        mgr = CheckpointManager(storage_dir=str(tmp_path))
        result = mgr.get_latest("any-session")
        assert result is None  # graceful failure

    def test_list_checkpoints_empty_db(self, tmp_path):
        """list_checkpoints on new DB returns empty list."""
        mgr = CheckpointManager(storage_dir=str(tmp_path))
        result = mgr.list_checkpoints("nonexistent")
        assert result == []


class TestStorageEdgeCases:
    def test_graph_with_very_long_labels(self, tmp_path):
        """Labels longer than 120 chars should be truncated, not crash."""
        g = GraphMemory("long-label", storage_dir=str(tmp_path))
        long_label = "Implement task with " + ("A" * 500)
        nid = g.add_node(NodeType.TASK, long_label)
        assert nid != ""
        assert len(g._nodes[nid].label) <= 120

    def test_graph_with_special_characters(self, tmp_path):
        """Unicode/special chars in labels should not break persistence."""
        g = GraphMemory("unicode-test", storage_dir=str(tmp_path))
        g.add_node(NodeType.TASK, "Implement 日本語 support with émojis 🚀")
        g._persist()

        g2 = GraphMemory("unicode-test", storage_dir=str(tmp_path))
        assert len(g2._nodes) == 1

    def test_many_sessions_isolated(self, tmp_path):
        """Different session IDs should have isolated graphs."""
        g1 = GraphMemory("session-A", storage_dir=str(tmp_path))
        g1.add_node(NodeType.TASK, "Implement task for session A")
        g1._persist()

        g2 = GraphMemory("session-B", storage_dir=str(tmp_path))
        g2.add_node(NodeType.TASK, "Implement task for session B")
        g2._persist()

        # Reload both
        g1r = GraphMemory("session-A", storage_dir=str(tmp_path))
        g2r = GraphMemory("session-B", storage_dir=str(tmp_path))

        labels_a = {n.label for n in g1r._nodes.values()}
        labels_b = {n.label for n in g2r._nodes.values()}

        assert "Implement task for session A" in labels_a
        assert "Implement task for session A" not in labels_b
        assert "Implement task for session B" in labels_b
        assert "Implement task for session B" not in labels_a
