"""
Regression tests for TM-19: prune()'s hard cap must actually be
enforceable, and its candidate scoring must be correctly ordered.

Three bugs:

1. The importance-only fallback tier (used when age-based pruning alone
   doesn't find enough candidates — e.g. a graph where everything was
   created recently) blanket-excluded ALL nodes of type DECISION,
   including SUPERSEDED/ARCHIVED/INVALIDATED ones. Only ACTIVE
   (COMPLETED) decisions are meant to be permanently protected — a graph
   dominated by dead/historical decisions could never be pruned down to
   max_nodes at all, making the "hard cap" not actually hard.

2. `candidates.sort()` ran BEFORE the fallback tier's `candidates.extend()`
   — the two tiers' entries were never merged into one sorted order, so
   `candidates[:to_prune]` could prune in a nonsensical sequence (all
   age-based candidates first regardless of score, fallback entries
   appended afterward with no relative ordering to the first group).

3. The fallback tier's exclusion set (`{nid for _, nid in candidates}`)
   was rebuilt on every iteration of the list comprehension that used
   it — O(n) work per item, O(n^2) total for the whole fallback pass.
"""
from __future__ import annotations

import time

import pytest

from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType


@pytest.fixture
def graph(tmp_path):
    return GraphMemory("prune-cap-test", storage_dir=str(tmp_path))


class TestFallbackTierCanPruneInactiveDecisions:

    def test_superseded_decisions_are_prunable_when_cap_requires_it(self, graph):
        """A graph dominated by SUPERSEDED decisions, all created recently
        (so age-based pruning alone finds nothing — max_age_days is huge),
        must still be prunable down toward the cap via the fallback tier.

        NOTE: labels must be genuinely distinct, not just a templated
        string varying by a trailing number — add_node's fuzzy same-
        decision merge (see decision_tracker.py::_is_same_decision) treats
        labels that differ only by a trailing digit as ~85% word overlap,
        well above its 82% merge threshold, and collapses them into ONE
        node. That's correct, intentional behavior elsewhere in this
        codebase (see test_graph.py's test_prune_preserves_decisions
        docstring for the same lesson) — it just means this fixture needs
        50 ACTUALLY different decisions to test pruning 50 real nodes."""
        topics = [
            "database", "cache", "auth", "web framework", "frontend",
            "orm", "deployment", "queue", "storage", "language",
            "runtime", "api style", "architecture", "testing",
            "backend platform", "auth provider", "payments", "observability",
            "state management", "package manager", "styling", "search",
        ]
        for i in range(50):
            nid = graph.add_node(
                NodeType.DECISION,
                f"Chose option {chr(65 + i % 26)}{i} for the {topics[i % len(topics)]} "
                f"layer of subsystem {i}",
                NodeStatus.SUPERSEDED, importance=0.2,
            )
            assert nid != ""
        # Active decisions and a goal that must survive regardless
        active_id = graph.add_node(NodeType.DECISION, "Use PostgreSQL for storage",
                                   NodeStatus.COMPLETED, importance=0.9)
        goal_id = graph.add_node(NodeType.GOAL, "Build the payments service")

        pruned = graph.prune(max_nodes=10, max_age_days=3650)  # age pruning finds nothing
        assert pruned > 0, (
            "prune() found zero candidates in a graph dominated by "
            "SUPERSEDED decisions — the hard cap cannot be enforced at all "
            "for this graph shape"
        )
        assert len(graph._nodes) <= 10 + 2, (  # +2 slack: active decision + goal always survive
            f"graph still has {len(graph._nodes)} nodes after pruning toward "
            f"a cap of 10 — SUPERSEDED decisions must be eligible for the "
            f"importance-only fallback tier"
        )
        assert active_id in graph._nodes, "the ACTIVE decision must never be pruned"
        assert goal_id in graph._nodes, "GOAL nodes must never be pruned"

    def test_active_decisions_are_never_pruned_even_by_fallback(self, graph):
        """The fallback tier must still protect ACTIVE (COMPLETED)
        decisions and GOAL/SCHEMA — only inactive decisions gained
        eligibility, not everything.

        NOTE: each label uses a DIFFERENT recognized topic bucket (see
        decision_tracker.py's taxonomy) specifically so none of them
        auto-supersede each other via the topic-collision behavior
        add_node already has (that's a separate, known issue — TM-09 —
        not what this test is about). Reusing one bucket, or using
        labels with no recognized topic, either collapses them via
        fuzzy dedup or triggers unrelated supersession, both of which
        would make this fixture assert nothing about pruning."""
        topics = [
            "PostgreSQL for the database", "Redis for the cache backend",
            "JWT for the auth mechanism", "bcrypt for password hashing",
            "FastAPI as the web framework", "React as the frontend",
            "SQLAlchemy as the orm", "Docker for deployment",
            "Kafka as the queue", "S3 for object storage",
            "Python as the language", "Node as the runtime",
            "REST for the api style", "a modular architecture",
            "pytest for testing", "Supabase as the backend platform",
            "Auth0 as the auth provider", "Stripe for payments",
            "Grafana for observability", "Redux for state management",
        ]
        assert len(topics) == 20
        for t in topics:
            graph.add_node(NodeType.DECISION, f"Use {t}",
                           NodeStatus.COMPLETED, importance=0.1)  # low importance, but ACTIVE
        graph.prune(max_nodes=2, max_age_days=3650)
        active_decisions = [n for n in graph._nodes.values()
                           if n.type == NodeType.DECISION and n.status == NodeStatus.COMPLETED]
        assert len(active_decisions) == 20, (
            "an active decision was pruned by the fallback tier — only "
            "inactive (superseded/archived/invalidated) decisions should "
            "ever become eligible there"
        )


class TestCombinedCandidatesAreSortedTogether:

    def test_lowest_scoring_candidate_is_pruned_first_across_tiers(self, graph):
        """Regression for the sort-before-extend bug: an age-based
        candidate with a HIGH score and a fallback-tier candidate with a
        LOWER score must be pruned in TRUE combined order (lowest first),
        not "all age-based candidates, then all fallback candidates"."""
        # Age-based candidate: old + high importance -> high score, should
        # survive longer than a low-importance fresh one.
        old_high_importance = graph.add_node(
            NodeType.TASK, "Old but still fairly important cleanup task",
            NodeStatus.COMPLETED, importance=0.9,
        )
        graph._nodes[old_high_importance].updated_at = time.time() - 100 * 86400

        # Fallback-tier candidate: fresh (so NOT caught by age-based scan)
        # but very low importance -> should be pruned before the
        # high-importance old one if scoring is truly combined.
        fresh_low_importance = graph.add_node(
            NodeType.TASK, "Trivial fresh task nobody cares about",
            NodeStatus.PENDING, importance=0.1,
        )

        # Pad with enough distinct nodes to force pruning down to 1.
        for i in range(10):
            nid = graph.add_node(NodeType.FILE, f"src/module_{i}.py", importance=0.5)
            graph._nodes[nid].updated_at = time.time() - 100 * 86400

        graph.prune(max_nodes=1, max_age_days=30)
        # This is the actual regression check: the fresh, low-importance
        # placeholder is the WORST candidate in the graph on any correct
        # combined-and-sorted scoring, so it must not survive while the
        # higher-value, high-importance aged task does. Under the old
        # sort-before-extend bug, the fallback tier's entries were
        # appended AFTER sorting and could survive out of true score
        # order — this assertion would have caught that.
        assert fresh_low_importance not in graph._nodes, (
            "the lowest-scoring candidate across both tiers survived "
            "pruning — fallback-tier entries are not being sorted "
            "together with age-based ones"
        )
        assert old_high_importance in graph._nodes, (
            "the highest-scoring candidate was pruned instead of kept"
        )


class TestPruneScalesWithoutQuadraticBlowup:
    """Functional (not timing-based) check that the fallback tier's
    exclusion-set fix still produces correct results at a size where the
    old O(n^2) comprehension would have been doing real (if not fatally
    slow) redundant work."""

    def test_large_graph_prunes_correctly(self, graph):
        for i in range(300):
            graph.add_node(NodeType.TASK, f"Do a distinct thing number {i} today",
                           NodeStatus.PENDING, importance=0.3)
        pruned = graph.prune(max_nodes=50, max_age_days=3650)
        assert pruned > 0
        assert len(graph._nodes) <= 55  # some slack for ties, but cap roughly honored
