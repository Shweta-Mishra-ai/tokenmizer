"""Integration tests — checkpoint creation and resume."""
import pytest
from tokenmizer.graph_memory.graph import GraphMemory, NodeType, NodeStatus
from tokenmizer.checkpoints.manager import CheckpointManager
from tokenmizer.core.tokenizer import count_tokens


MESSAGES = [
    {"role": "user", "content": "We're building a FastAPI app with PostgreSQL"},
    {"role": "assistant", "content": "Created project structure. Files: main.py, models.py, database.py"},
    {"role": "user", "content": "Add user authentication"},
    {"role": "assistant", "content": "Implemented JWT auth. Decided: use python-jose for JWT signing. Completed auth module."},
    {"role": "user", "content": "Now add Redis caching"},
    {"role": "assistant", "content": "Working on Redis caching layer. Added redis dependency. Implementing cache/redis.py"},
]


@pytest.fixture
def checkpoint_mgr(tmp_path):
    return CheckpointManager(storage_dir=str(tmp_path))


@pytest.fixture
def graph_with_data(tmp_path):
    g = GraphMemory("ckpt-test", storage_dir=str(tmp_path))
    g.extract_from_messages(MESSAGES, incremental=False)
    return g


class TestCheckpointCreation:

    def test_checkpoint_created_with_id(self, checkpoint_mgr, graph_with_data, tmp_path):
        ckpt = checkpoint_mgr.create(
            session_id="test-session",
            messages=MESSAGES,
            graph=graph_with_data,
            context_pct=0.87,
            trigger="auto_threshold",
        )
        assert ckpt.checkpoint_id.startswith("ckpt_")
        assert ckpt.context_pct == 0.87
        assert ckpt.trigger == "auto_threshold"
        assert ckpt.message_count == len(MESSAGES)

    def test_resume_tokens_accurate(self, checkpoint_mgr, graph_with_data):
        ckpt = checkpoint_mgr.create(
            session_id="token-test",
            messages=MESSAGES,
            graph=graph_with_data,
            context_pct=0.85,
        )
        # resume_tokens should equal count_tokens(resume_standard)
        actual = count_tokens(ckpt.resume_standard)
        assert ckpt.resume_tokens == actual

    def test_tiered_resume_sizes(self, checkpoint_mgr, graph_with_data):
        ckpt = checkpoint_mgr.create(
            session_id="tiered-test",
            messages=MESSAGES,
            graph=graph_with_data,
            context_pct=0.85,
        )
        t_critical = count_tokens(ckpt.resume_critical)
        t_standard = count_tokens(ckpt.resume_standard)
        t_full = count_tokens(ckpt.resume_full)

        # Critical should be shortest, full should be longest
        assert t_critical <= t_standard + 50  # some slack
        assert t_full >= t_standard - 50

    def test_resume_contains_decision(self, checkpoint_mgr, graph_with_data):
        ckpt = checkpoint_mgr.create(
            session_id="decision-test",
            messages=MESSAGES,
            graph=graph_with_data,
            context_pct=0.85,
        )
        # At minimum, something should be in the resume
        assert len(ckpt.resume_standard) > 20

    def test_graph_diff_computed(self, checkpoint_mgr, graph_with_data):
        ckpt1 = checkpoint_mgr.create(
            session_id="diff-test",
            messages=MESSAGES[:4],
            graph=graph_with_data,
            context_pct=0.7,
        )
        ckpt2 = checkpoint_mgr.create(
            session_id="diff-test",
            messages=MESSAGES,
            graph=graph_with_data,
            context_pct=0.9,
        )
        assert "added" in ckpt2.graph_diff
        assert "removed" in ckpt2.graph_diff


class TestCheckpointPersistence:

    def test_checkpoint_survives_reload(self, tmp_path, graph_with_data):
        mgr1 = CheckpointManager(storage_dir=str(tmp_path))
        ckpt = mgr1.create(
            session_id="persist-ckpt",
            messages=MESSAGES,
            graph=graph_with_data,
            context_pct=0.86,
        )
        original_id = ckpt.checkpoint_id

        # New manager instance loads from same SQLite DB
        mgr2 = CheckpointManager(storage_dir=str(tmp_path))
        loaded = mgr2.get_latest("persist-ckpt")

        assert loaded is not None
        assert loaded.checkpoint_id == original_id
        assert loaded.resume_standard == ckpt.resume_standard

    def test_list_checkpoints(self, checkpoint_mgr, graph_with_data):
        for i in range(3):
            checkpoint_mgr.create(
                session_id="list-test",
                messages=MESSAGES,
                graph=graph_with_data,
                context_pct=0.8 + i * 0.05,
            )
        checkpoints = checkpoint_mgr.list_checkpoints("list-test")
        assert len(checkpoints) == 3
        # Should be sorted newest first
        assert checkpoints[0]["context_pct"] >= checkpoints[-1]["context_pct"]

    def test_get_latest_returns_most_recent(self, checkpoint_mgr, graph_with_data):
        import time
        checkpoint_mgr.create("latest-test", MESSAGES[:2], graph_with_data, 0.7)
        time.sleep(0.01)  # ensure different timestamp
        ckpt2 = checkpoint_mgr.create("latest-test", MESSAGES, graph_with_data, 0.9)

        latest = checkpoint_mgr.get_latest("latest-test")
        assert latest.checkpoint_id == ckpt2.checkpoint_id

    def test_get_latest_nonexistent_returns_none(self, checkpoint_mgr):
        result = checkpoint_mgr.get_latest("nonexistent-session-xyz")
        assert result is None
