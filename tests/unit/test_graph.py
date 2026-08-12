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
        graph.add_node(NodeType.TASK, "Implement auth", NodeStatus.PENDING)
        assert graph._nodes[id1].status == NodeStatus.COMPLETED

    def test_labels_identical_after_truncation_share_node_id(self, graph):
        label = "Implement authentication " + "A" * 150

        id1 = graph.add_node(NodeType.TASK, label)
        id2 = graph.add_node(NodeType.TASK, label[:120])

        assert id1 == id2
        assert len(graph._nodes) == 1
        assert graph._nodes[id1].label == label[:120]


class TestErrorDeduplication:
    """Phase 3 of the memory-improvement plan: ERROR nodes get a fuzzy
    near-duplicate merge (like DECISION already had), plus a real bug
    fix found while building it — see _next_status() in graph.py.

    Bug: the shared status-upgrade-only rule ranked FAILED (3) above
    COMPLETED (2) — correct for DECISION/TASK, where a re-mention must
    never regress an already-advanced node, but wrong for ERROR: a
    "this is fixed now" re-add (NodeStatus.COMPLETED) of an error
    already recorded as FAILED could never win (2 is not > 3), so the
    graph reported a resolved bug as open forever. No test caught this
    before — the first two tests below are that regression test.
    """

    def test_resolved_re_add_now_actually_resolves_it(self, graph):
        id1 = graph.add_node(NodeType.ERROR, "Connection to DB times out",
                             NodeStatus.FAILED, importance=0.9)
        id2 = graph.add_node(NodeType.ERROR, "Connection to DB times out",
                             NodeStatus.COMPLETED, importance=0.5)
        assert id1 == id2
        assert graph._nodes[id1].status == NodeStatus.COMPLETED, (
            f"a 'fixed now' re-add must resolve the error, got {graph._nodes[id1].status}"
        )

    def test_a_fixed_error_can_reopen(self, graph):
        """The reverse direction matters too: a bug that regresses must
        be able to go from COMPLETED back to FAILED, not get stuck
        showing as resolved forever."""
        id1 = graph.add_node(NodeType.ERROR, "Connection to DB times out",
                             NodeStatus.COMPLETED, importance=0.5)
        id2 = graph.add_node(NodeType.ERROR, "Connection to DB times out",
                             NodeStatus.FAILED, importance=0.9)
        assert id1 == id2
        assert graph._nodes[id1].status == NodeStatus.FAILED

    def test_decision_status_still_only_upgrades(self, graph):
        """The _next_status refactor must not change DECISION/TASK
        behavior — only ERROR gets the latest-wins rule."""
        id1 = graph.add_node(NodeType.DECISION, "Use PostgreSQL for storage",
                             NodeStatus.COMPLETED)
        graph.add_node(NodeType.DECISION, "Use PostgreSQL for storage",
                       NodeStatus.PENDING)
        assert graph._nodes[id1].status == NodeStatus.COMPLETED, (
            "a stale re-mention must not regress an already-COMPLETED decision"
        )

    def test_near_duplicate_error_merges_lexically(self, graph):
        """Containment: 'Connection timeout' inside a longer restatement
        merges without any embedding model — same rule _is_same_decision
        already used for decisions, applied to errors."""
        id1 = graph.add_node(NodeType.ERROR, "Connection timeout",
                             NodeStatus.FAILED, importance=0.9)
        id2 = graph.add_node(
            NodeType.ERROR, "Connection timeout on retry attempt 3",
            NodeStatus.FAILED, importance=0.9,
        )
        assert id1 == id2
        assert sum(1 for n in graph._nodes.values() if n.type == NodeType.ERROR) == 1

    def test_named_exception_classes_never_merge_despite_high_overlap(self, graph):
        """'TypeError: x is not a function' and 'ReferenceError: x is not
        a function' share every word except the exception class — high
        enough overlap to clear the containment/0.82 threshold, but they
        are genuinely different failures. See
        _named_error_classes_conflict in decision_tracker.py."""
        id1 = graph.add_node(NodeType.ERROR, "TypeError: x is not a function",
                             NodeStatus.FAILED, importance=0.9)
        id2 = graph.add_node(NodeType.ERROR, "ReferenceError: x is not a function",
                             NodeStatus.FAILED, importance=0.9)
        assert id1 != id2
        assert sum(1 for n in graph._nodes.values() if n.type == NodeType.ERROR) == 2


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

    def test_add_node_threads_source_role_to_validator(self, tmp_path):
        """Regression test for TM-29: add_node() never passed source_role
        to validator.validate() at all, so every node got the "assistant"
        confidence bonus regardless of the caller's actual knowledge of
        who stated it. Uses the default confidence=0.7 sentinel so the
        stored node confidence is the validator's OWN computed score
        (see add_node's `confidence if confidence != 0.7 else
        result.confidence`) rather than an explicit override — that's
        the only way this wiring is externally observable."""
        g_user = GraphMemory("role-wiring-user", storage_dir=str(tmp_path / "user"))
        nid_user = g_user.add_node(
            NodeType.DECISION, "Use MongoDB for the catalog service",
            NodeStatus.COMPLETED, source_role="user",
        )
        g_asst = GraphMemory("role-wiring-assistant", storage_dir=str(tmp_path / "asst"))
        nid_asst = g_asst.add_node(
            NodeType.DECISION, "Use MongoDB for the catalog service",
            NodeStatus.COMPLETED, source_role="assistant",
        )
        assert nid_user and nid_asst, "both variants should be accepted"
        assert g_asst._nodes[nid_asst].confidence > g_user._nodes[nid_user].confidence, (
            f"assistant-sourced confidence {g_asst._nodes[nid_asst].confidence} should "
            f"exceed user-sourced confidence {g_user._nodes[nid_user].confidence} — "
            f"equal values mean add_node() isn't actually passing source_role through"
        )

    def test_apply_extracted_forwards_decision_source_role_to_add_node(self, graph, monkeypatch):
        """_apply_extracted() must forward the per-item "source_role" a
        heuristic-extracted decision dict carries through to add_node() —
        this is the link between HybridExtractor attaching the real role
        (test_hybrid_extractor.py) and add_node() actually using it
        (test_add_node_threads_source_role_to_validator above). Verified
        directly via monkeypatch since the stored node confidence alone
        can't distinguish this — _apply_extracted passes an explicit
        confidence override for decisions, which masks the validator's
        own score in the final stored value."""
        captured: list[dict] = []
        real_add_node = graph.add_node

        def _spy_add_node(*args, **kwargs):
            captured.append(kwargs)
            return real_add_node(*args, **kwargs)

        monkeypatch.setattr(graph, "add_node", _spy_add_node)
        graph._apply_extracted(
            {"decisions": [{"label": "Use MongoDB for the catalog service",
                            "reason": "", "source_role": "user"}]},
            messages=[],
        )
        decision_calls = [kw for kw in captured if kw.get("source_role") is not None]
        assert decision_calls, f"add_node was never called with source_role set: {captured}"
        assert decision_calls[0]["source_role"] == "user"


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
        # Ten genuinely distinct decisions in DIFFERENT topic buckets.
        # (The old fixture used "Use SQLite for local storage {i}" — labels
        # differing only by a trailing digit are 83% word-overlap, which
        # _is_same_decision has always considered the same decision; since
        # the 2026-07-10 near-duplicate merge fix, add_node now acts on
        # that judgment and merges them, so the fixture must use decisions
        # that are actually distinct.)
        labels = [
            "Use PostgreSQL for the primary database",
            "Use Redis as the cache backend",
            "Use JWT tokens for the auth mechanism",
            "Use bcrypt for password hashing",
            "Use FastAPI as the web framework",
            "Use React for the frontend framework",
            "Deploy on Docker with compose",
            "Use Kafka as the message queue",
            "Use S3 for object file storage",
            "Use pytest as the testing framework",
        ]
        for label in labels:
            nid = graph.add_node(NodeType.DECISION, label, NodeStatus.COMPLETED)
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


class TestArchivedReachability:
    """
    SUPERSEDED decisions age into ARCHIVED after
    GraphMemory.ARCHIVE_SUPERSEDED_AFTER_DAYS, measured from supersession
    time. apply_importance_decay() is the only path that sets ARCHIVED,
    so these tests guard the state's reachability.
    """

    def test_superseded_decision_ages_into_archived(self, graph):
        import time as _time
        old_id = graph.add_node(NodeType.DECISION, "Use PostgreSQL for storage",
                                NodeStatus.COMPLETED)
        graph.add_node(NodeType.DECISION, "Switch to MySQL for storage",
                       NodeStatus.COMPLETED)
        old = graph._nodes[old_id]
        assert old.status == NodeStatus.SUPERSEDED, "precondition: supersession fired"

        # Simulate 8 days passing since supersession
        old.valid_until = _time.time() - 8 * 86400
        graph.apply_importance_decay()
        assert old.status == NodeStatus.ARCHIVED

    def test_recently_superseded_decision_not_archived(self, graph):
        old_id = graph.add_node(NodeType.DECISION, "Use PostgreSQL for storage",
                                NodeStatus.COMPLETED)
        graph.add_node(NodeType.DECISION, "Switch to MySQL for storage",
                       NodeStatus.COMPLETED)
        graph.apply_importance_decay()
        assert graph._nodes[old_id].status == NodeStatus.SUPERSEDED

    def test_active_decisions_never_archived(self, graph):
        nid = graph.add_node(NodeType.DECISION, "Use PostgreSQL for storage",
                             NodeStatus.COMPLETED)
        graph.apply_importance_decay()
        assert graph._nodes[nid].status == NodeStatus.COMPLETED


class TestStatsExcludesEvictedNodes:
    """Regression test for TM-35: stats() was the one place in the
    codebase that didn't filter out _evicted nodes — query(),
    to_context_block(), and both visualization.py exporters all do.
    A caller comparing /api/graph/{id} (stats) against /api/graph/{id}/viz
    (visualization) would see disagreeing node counts."""

    def test_evicted_node_excluded_from_node_count(self, graph):
        nid = graph.add_node(NodeType.TASK, "Implement the checkout flow")
        graph.add_node(NodeType.TASK, "Implement the login flow")
        graph._nodes[nid]._evicted = True

        stats = graph.stats()
        assert stats["node_count"] == 1
        assert stats["by_type"]["task"] == 1

    def test_evicted_node_excluded_from_by_status(self, graph):
        nid = graph.add_node(NodeType.TASK, "Implement the checkout flow",
                             NodeStatus.COMPLETED)
        graph.add_node(NodeType.TASK, "Implement the login flow", NodeStatus.PENDING)
        graph._nodes[nid]._evicted = True

        stats = graph.stats()
        assert stats["by_status"].get("completed", 0) == 0
        assert stats["by_status"]["pending"] == 1
