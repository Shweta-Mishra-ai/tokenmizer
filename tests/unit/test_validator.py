"""Unit tests — graph validator and confidence scoring."""
import pytest

from tokenmizer.graph_memory.validator import GraphValidator


@pytest.fixture
def v():
    return GraphValidator(min_confidence=0.50)


class TestHardRejects:

    def test_empty_label_rejected(self, v):
        r = v.validate("", "task")
        assert r.accepted is False

    def test_noise_word_rejected(self, v):
        for word in ["this", "that", "it", "ok", "done", "yes"]:
            r = v.validate(word, "task")
            assert r.accepted is False, f"'{word}' should be rejected"

    def test_pure_number_rejected(self, v):
        r = v.validate("42", "task")
        assert r.accepted is False

    def test_too_short_rejected(self, v):
        r = v.validate("ab", "task")
        assert r.accepted is False

    def test_generic_single_verb_rejected(self, v):
        r = v.validate("implement", "task")
        assert r.accepted is False

    def test_url_rejected(self, v):
        r = v.validate("https://example.com/api", "task")
        assert r.accepted is False


class TestAccepted:

    def test_specific_task_accepted(self, v):
        r = v.validate("Implement JWT refresh token rotation", "task")
        assert r.accepted is True
        assert r.confidence >= 0.55

    def test_file_with_extension_accepted(self, v):
        r = v.validate("api/auth.py", "file")
        assert r.accepted is True
        assert r.confidence >= 0.70

    def test_decision_with_rationale_accepted(self, v):
        r = v.validate(
            "Use Redis for session storage",
            "decision",
            summary="Faster than PostgreSQL for ephemeral data",
        )
        assert r.accepted is True
        assert r.confidence >= 0.70

    def test_decision_without_rationale_lower_confidence(self, v):
        with_rationale = v.validate("Use Redis", "decision", summary="faster than postgres")
        without_rationale = v.validate("Use Redis", "decision", summary="")
        assert with_rationale.confidence > without_rationale.confidence

    def test_environment_with_version_accepted(self, v):
        r = v.validate("Python 3.12", "environment")
        assert r.accepted is True
        assert r.confidence >= 0.70

    def test_dependency_with_version_accepted(self, v):
        r = v.validate("fastapi>=0.111.0", "dependency")
        assert r.accepted is True
        assert r.confidence >= 0.70

    def test_goal_accepted(self, v):
        r = v.validate("Build FastAPI authentication service with JWT", "goal")
        assert r.accepted is True

    def test_error_accepted(self, v):
        r = v.validate("422 Unprocessable Entity on login endpoint", "error")
        assert r.accepted is True


class TestConfidenceScoring:

    def test_longer_label_higher_confidence(self, v):
        short = v.validate("Fix bug", "task")
        long = v.validate("Fix authentication 422 error in login endpoint", "task")
        if short.accepted and long.accepted:
            assert long.confidence >= short.confidence

    def test_summary_boosts_confidence(self, v):
        no_summary = v.validate("Use PostgreSQL", "decision", summary="")
        with_summary = v.validate("Use PostgreSQL", "decision",
                                  summary="needed for concurrent writes in production")
        if no_summary.accepted and with_summary.accepted:
            assert with_summary.confidence > no_summary.confidence

    def test_confidence_bounded_0_to_1(self, v):
        for label, node_type in [
            ("Implement auth", "task"),
            ("api/main.py", "file"),
            ("Use Redis", "decision"),
        ]:
            r = v.validate(label, node_type)
            assert 0.0 <= r.confidence <= 1.0


class TestTypeMismatch:

    def test_file_path_as_task_gets_corrected(self, v):
        r = v.validate("api/auth.py", "task")
        # Should either reject or correct type to "file"
        if r.accepted:
            assert r.corrected_type == "file"

    def test_endpoint_as_task_gets_corrected(self, v):
        r = v.validate("POST /api/auth/login", "task")
        if r.accepted:
            assert r.corrected_type == "endpoint"

    def test_dep_pattern_gets_corrected(self, v):
        r = v.validate("fastapi==0.111.0", "task")
        if r.accepted:
            assert r.corrected_type == "dependency"


class TestGraphIntegration:
    """Test validator wired into GraphMemory.add_node()."""

    def test_noise_node_not_added(self, tmp_path):
        from tokenmizer.graph_memory.graph import GraphMemory, NodeType
        g = GraphMemory("val-test", storage_dir=str(tmp_path))
        nid = g.add_node(NodeType.TASK, "this")
        assert nid == ""  # rejected
        assert len(g._nodes) == 0

    def test_good_node_added_with_confidence(self, tmp_path):
        from tokenmizer.graph_memory.graph import GraphMemory, NodeType
        g = GraphMemory("val-test2", storage_dir=str(tmp_path))
        nid = g.add_node(NodeType.TASK, "Implement JWT authentication middleware")
        assert nid != ""
        assert len(g._nodes) == 1
        node = g._nodes[nid]
        assert node.confidence > 0.0
        assert node.confidence <= 1.0

    def test_rejected_nodes_not_in_stats(self, tmp_path):
        from tokenmizer.graph_memory.graph import GraphMemory, NodeType
        g = GraphMemory("val-test3", storage_dir=str(tmp_path))
        g.add_node(NodeType.TASK, "this")   # rejected
        g.add_node(NodeType.TASK, "that")   # rejected
        g.add_node(NodeType.TASK, "Implement refresh token rotation")  # accepted
        stats = g.stats()
        assert stats["node_count"] == 1

    def test_avg_confidence_in_stats(self, tmp_path):
        from tokenmizer.graph_memory.graph import GraphMemory, NodeType
        g = GraphMemory("val-test4", storage_dir=str(tmp_path))
        g.add_node(NodeType.TASK, "Implement auth endpoint")
        g.add_node(NodeType.DECISION, "Use PostgreSQL for storage",
                   summary="concurrent writes needed")
        stats = g.stats()
        assert "avg_confidence" in stats
        assert 0.0 < stats["avg_confidence"] <= 1.0

    def test_semantic_edges_not_accidental(self, tmp_path):
        from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType
        g = GraphMemory("edge-test", storage_dir=str(tmp_path))

        # Add task about auth
        t1 = g.add_node(NodeType.TASK, "Implement auth middleware", NodeStatus.IN_PROGRESS)
        # Add unrelated task about database
        t2 = g.add_node(NodeType.TASK, "Set up PostgreSQL connection pooling", NodeStatus.IN_PROGRESS)
        # Add file for auth
        f1 = g.add_node(NodeType.FILE, "api/auth.py")

        if t1 and t2 and f1:
            # auth.py should be linked to auth task but NOT to postgres task
            auth_task_node = g._nodes.get(t1)
            edges = [(e.source_id, e.target_id) for e in g._edges]
            if auth_task_node:
                # The postgres task should not have an edge to auth.py
                assert (t2, f1) not in edges, \
                    "PostgreSQL task should NOT be linked to api/auth.py"
