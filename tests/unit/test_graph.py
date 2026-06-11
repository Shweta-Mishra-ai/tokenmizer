"""Unit tests — graph memory."""

import pytest

from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType


@pytest.fixture
def graph(tmp_path):
    return GraphMemory("test-session", storage_dir=str(tmp_path))


class TestNodeDeduplication:

    def test_duplicate_node_not_added(self, graph):
        id1 = graph.add_node(NodeType.TASK, "Implement auth")
        id2 = graph.add_node(NodeType.TASK, "Implement auth")
        assert id1 == id2
        assert len(graph._nodes) == 1

    def test_duplicate_case_insensitive(self, graph):
        id1 = graph.add_node(NodeType.TASK, "Implement Auth")
        id2 = graph.add_node(NodeType.TASK, "implement auth")
        assert id1 == id2
        assert len(graph._nodes) == 1

    def test_duplicate_trailing_punctuation(self, graph):
        id1 = graph.add_node(NodeType.TASK, "Implement auth.")
        id2 = graph.add_node(NodeType.TASK, "Implement auth")
        assert id1 == id2

    def test_different_type_same_label_different_nodes(self, graph):
        id1 = graph.add_node(NodeType.TASK, "Implement auth")
        id2 = graph.add_node(NodeType.FILE, "api/auth.py")
        assert id1 != id2
        assert len(graph._nodes) == 2

    def test_status_upgrades_on_duplicate(self, graph):
        id1 = graph.add_node(NodeType.TASK, "Implement auth", NodeStatus.PENDING)
        id2 = graph.add_node(NodeType.TASK, "Implement auth", NodeStatus.COMPLETED)
        assert id1 == id2
        assert graph._nodes[id1].status == NodeStatus.COMPLETED

    def test_status_does_not_downgrade(self, graph):
        id1 = graph.add_node(NodeType.TASK, "Implement auth", NodeStatus.COMPLETED)
        id2 = graph.add_node(NodeType.TASK, "Implement auth", NodeStatus.PENDING)
        assert graph._nodes[id1].status == NodeStatus.COMPLETED


class TestNodeTypes:

    def test_all_new_node_types_accepted(self, graph):
        labels = {
            NodeType.ENVIRONMENT: "Python 3.12",
            NodeType.GOAL: "Build FastAPI authentication service",
            NodeType.TEST: "run pytest auth suite",
            NodeType.ENDPOINT: "POST /api/auth/login",
            NodeType.SCHEMA: "User: id, email, password_hash",
        }
        for nt, label in labels.items():
            nid = graph.add_node(nt, label)
            assert nid != ""
            assert nid in graph._nodes

    def test_goal_gets_high_importance(self, graph):
        nid = graph.add_node(NodeType.GOAL, "Build auth service", importance=1.0)
        assert graph._nodes[nid].importance == 1.0


class TestExtraction:

    def test_extract_from_messages_basic(self, graph):
        messages = [
            {"role": "user", "content": "Let's build a FastAPI auth service"},
            {"role": "assistant", "content": "I completed setting up the project structure. Created api/main.py and api/auth.py"},
        ]
        graph.extract_from_messages(messages, incremental=False)
        assert len(graph._nodes) > 0

    def test_files_extracted(self, graph):
        messages = [
            {"role": "assistant", "content": "I created api/auth.py and updated config.py"},
        ]
        graph.extract_from_messages(messages, incremental=False)
        file_nodes = [n for n in graph._nodes.values() if n.type == NodeType.FILE]
        assert len(file_nodes) >= 1

    def test_incremental_skips_processed(self, graph):
        messages = [{"role": "user", "content": "Build auth.py"}]
        graph.extract_from_messages(messages, incremental=True)
        count_after_first = len(graph._nodes)

        # Same messages again — should be skipped
        graph.extract_from_messages(messages, incremental=True)
        assert len(graph._nodes) == count_after_first

    def test_secrets_redacted_in_nodes(self, graph):
        messages = [
            {"role": "user", "content": "My API key is sk-ant-api03-secret123456789abcdef"},
        ]
        graph.extract_from_messages(messages, incremental=False)
        for node in graph._nodes.values():
            assert "sk-ant" not in node.label
            assert "sk-ant" not in node.summary


class TestPruning:

    def test_prune_removes_old_completed_nodes(self, graph):
        import time
        # Add many completed tasks with old timestamp
        for i in range(10):
            nid = graph.add_node(NodeType.TASK, f"Implement old task {i}", NodeStatus.COMPLETED)
            assert nid != ""
            graph._nodes[nid].updated_at = time.time() - 100 * 86400  # 100 days ago

        # Add some recent ones
        for i in range(5):
            nid = graph.add_node(NodeType.TASK, f"Implement recent task {i}", NodeStatus.PENDING)
            assert nid != ""

        pruned = graph.prune(max_nodes=5, max_age_days=30)
        assert pruned > 0
        assert len(graph._nodes) <= 10  # pruned some

    def test_prune_preserves_decisions(self, graph):
        import time
        for i in range(10):
            nid = graph.add_node(NodeType.DECISION, f"Use SQLite for local storage {i}", NodeStatus.COMPLETED)
            assert nid != ""
            graph._nodes[nid].updated_at = time.time() - 100 * 86400

        graph.prune(max_nodes=2, max_age_days=150)
        decisions = [n for n in graph._nodes.values() if n.type == NodeType.DECISION]
        assert len(decisions) == 10  # all preserved

    def test_prune_preserves_goals(self, graph):
        import time
        for i in range(5):
            nid = graph.add_node(NodeType.GOAL, f"Build FastAPI authentication service {i}")
            assert nid != ""
            graph._nodes[nid].updated_at = time.time() - 100 * 86400

        graph.prune(max_nodes=1, max_age_days=1)
        goals = [n for n in graph._nodes.values() if n.type == NodeType.GOAL]
        assert len(goals) == 5


class TestContextBlock:

    def test_context_block_respects_budget(self, graph):
        from tokenmizer.core.tokenizer import count_tokens
        # Fill graph with lots of data
        for i in range(50):
            graph.add_node(NodeType.TASK, f"Implement task number {i} doing something important")
        for i in range(20):
            graph.add_node(NodeType.DECISION, f"Use SQLite database {i} for storage")

        block = graph.to_context_block(token_budget=200)
        tokens = count_tokens(block)
        assert tokens <= 250  # some slack for edge cases

    def test_context_block_includes_open_tasks(self, graph):
        graph.add_node(NodeType.TASK, "Implement login endpoint", NodeStatus.IN_PROGRESS)
        block = graph.to_context_block()
        assert "login endpoint" in block.lower() or "implement" in block.lower()


class TestPersistence:

    def test_graph_survives_reload(self, tmp_path):
        g1 = GraphMemory("persist-test", storage_dir=str(tmp_path))
        g1.add_node(NodeType.TASK, "Implement auth", NodeStatus.COMPLETED)
        g1.add_node(NodeType.DECISION, "Use JWT tokens", summary="stateless auth")
        g1._persist()

        # New instance — loads from disk
        g2 = GraphMemory("persist-test", storage_dir=str(tmp_path))
        assert len(g2._nodes) == 2
        task_node = next(n for n in g2._nodes.values() if n.type == NodeType.TASK)
        assert task_node.label == "Implement auth"
        assert task_node.status == NodeStatus.COMPLETED
