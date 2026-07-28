"""
Regression tests for TM-09: coarse topic-bucket collision must not
silently supersede two decisions that are actually complementary, not a
reversal of each other.

Bug: find_contradicting_decisions() superseded ANY active decision
sharing a topic bucket with a new one — "Use PostgreSQL for primary user
data" and "Use SQLite for the local offline cache" both map to the
"database" topic bucket, so adding the second silently marked the first
SUPERSEDED, excluding it from query()/to_context_block() as if it had
been replaced. Reproduced in the audit.

Fix: topic-bucket overlap is now necessary but not sufficient. An
additional "slot" check compares the DESCRIPTIVE words in each label
(everything except topic-taxonomy keywords and stopwords) — genuine
same-purpose replacements ("Use PostgreSQL for storage" -> "Use MySQL for
storage") still supersede as before, but decisions whose descriptive
context clearly differs ("primary user data" vs "local offline cache")
are instead marked CONTESTED on both sides — visible, flagged, and left
for explicit resolution rather than one being silently destroyed on
ambiguous evidence.
"""
from __future__ import annotations

from tokenmizer.graph_memory.decision_tracker import find_contradicting_decisions
from tokenmizer.graph_memory.graph import EdgeType, GraphMemory, NodeStatus, NodeType


def _graph(tmp_path, session_id="t-contest"):
    return GraphMemory(session_id=session_id, storage_dir=str(tmp_path))


class TestComplementaryDecisionsAreNotSuperseded:

    def test_different_purpose_same_topic_does_not_supersede(self, tmp_path):
        g = _graph(tmp_path)
        old_id = g.add_node(NodeType.DECISION, "Use PostgreSQL for primary user data",
                            NodeStatus.COMPLETED, summary="relational integrity matters")
        hits = find_contradicting_decisions(
            "Use SQLite for the local offline cache", "zero-config embedded", g._nodes,
        )
        assert old_id not in hits, (
            "two decisions about different purposes within the same topic "
            "bucket (primary datastore vs local offline cache) must not "
            "supersede each other just because both are 'database'"
        )

    def test_both_sides_become_contested_via_add_node(self, tmp_path):
        g = _graph(tmp_path)
        old_id = g.add_node(NodeType.DECISION, "Use PostgreSQL for primary user data",
                            NodeStatus.COMPLETED, summary="relational integrity matters")
        new_id = g.add_node(NodeType.DECISION, "Use SQLite for the local offline cache",
                            NodeStatus.COMPLETED, summary="zero-config embedded")

        assert g._nodes[old_id].status == NodeStatus.CONTESTED, (
            f"expected CONTESTED, got {g._nodes[old_id].status}"
        )
        assert g._nodes[new_id].status == NodeStatus.CONTESTED, (
            f"expected CONTESTED, got {g._nodes[new_id].status}"
        )

    def test_contested_pair_gets_conflicts_with_edge(self, tmp_path):
        g = _graph(tmp_path)
        old_id = g.add_node(NodeType.DECISION, "Use PostgreSQL for primary user data",
                            NodeStatus.COMPLETED, summary="relational integrity matters")
        new_id = g.add_node(NodeType.DECISION, "Use SQLite for the local offline cache",
                            NodeStatus.COMPLETED, summary="zero-config embedded")
        conflict_edges = [
            e for e in g._edges
            if e.type == EdgeType.CONFLICTS_WITH
            and {e.source_id, e.target_id} == {old_id, new_id}
        ]
        assert conflict_edges, "expected a CONFLICTS_WITH edge between the contested pair"

    def test_contested_decisions_do_not_create_a_transition(self, tmp_path):
        """Unlike genuine supersession, an unresolved conflict is not a
        causal transition — no DecisionTransition record should exist for
        a CONTESTED pair."""
        g = _graph(tmp_path)
        g.add_node(NodeType.DECISION, "Use PostgreSQL for primary user data",
                  NodeStatus.COMPLETED, summary="relational integrity matters")
        g.add_node(NodeType.DECISION, "Use SQLite for the local offline cache",
                  NodeStatus.COMPLETED, summary="zero-config embedded")
        assert g.get_transitions() == []


class TestGenuineReplacementsStillSupersede:
    """Guard against the fix being so conservative it stops catching
    real reversals — the existing, simpler "Use X" -> "Use Y" case must
    keep working exactly as before."""

    def test_same_purpose_swap_still_supersedes(self, tmp_path):
        g = _graph(tmp_path)
        old_id = g.add_node(NodeType.DECISION, "Use PostgreSQL for storage",
                            NodeStatus.COMPLETED)
        new_id = g.add_node(NodeType.DECISION, "Use MySQL for storage",
                            NodeStatus.COMPLETED)
        assert g._nodes[old_id].status == NodeStatus.SUPERSEDED
        assert g._nodes[new_id].status == NodeStatus.COMPLETED
        assert len(g.get_transitions()) == 1

    def test_bare_tech_swap_with_no_descriptive_context_still_supersedes(self, tmp_path):
        g = _graph(tmp_path)
        old_id = g.add_node(NodeType.DECISION, "Use Supabase for the backend",
                            NodeStatus.COMPLETED)
        new_id = g.add_node(NodeType.DECISION, "Switch to Firebase",
                            NodeStatus.COMPLETED)
        assert g._nodes[old_id].status == NodeStatus.SUPERSEDED
        assert g._nodes[new_id].status == NodeStatus.COMPLETED


class TestContestedNodesRemainVisible:

    def test_query_returns_contested_nodes(self, tmp_path):
        g = _graph(tmp_path)
        old_id = g.add_node(NodeType.DECISION, "Use PostgreSQL for primary user data",
                            NodeStatus.COMPLETED, summary="relational integrity matters",
                            importance=0.9)
        new_id = g.add_node(NodeType.DECISION, "Use SQLite for the local offline cache",
                            NodeStatus.COMPLETED, summary="zero-config embedded",
                            importance=0.9)
        results = g.query("database storage decision", top_k=10)
        result_ids = {n.id for n in results}
        assert old_id in result_ids or new_id in result_ids, (
            "CONTESTED decisions must remain reachable via query() — "
            "unlike SUPERSEDED, they represent live, unresolved information"
        )

    def test_context_block_surfaces_the_conflict(self, tmp_path):
        g = _graph(tmp_path)
        g.add_node(NodeType.DECISION, "Use PostgreSQL for primary user data",
                  NodeStatus.COMPLETED, summary="relational integrity matters",
                  importance=0.9)
        g.add_node(NodeType.DECISION, "Use SQLite for the local offline cache",
                  NodeStatus.COMPLETED, summary="zero-config embedded", importance=0.9)
        block = g.to_context_block(token_budget=2000)
        assert "postgresql" in block.lower() and "sqlite" in block.lower(), (
            f"expected both contested decisions to be visible in resume "
            f"context, got:\n{block}"
        )
