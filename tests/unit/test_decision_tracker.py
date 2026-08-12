"""
Tests for the decision topic classifier and same-decision detection.

Covers the known-hard cases: the imperative "Go with X" vs. Go-the-language
ambiguity, technology names shared with the extractor vocabulary, multi-topic
statements, bigram/single-word precedence, and near-duplicate label
containment.
"""
from tokenmizer.graph_memory.decision_tracker import (
    classify_topic,
    classify_topics,
    find_contradicting_decisions,
)
from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType

# ── classify_topics: the audit cases ─────────────────────────────────────────

def test_go_verb_does_not_classify_as_language():
    """'Go with X' is an imperative, not a language choice."""
    topics = classify_topics("Go with tRPC for the API layer")
    assert "language" not in topics
    assert "api_style" in topics


def test_go_language_still_detected_with_context():
    assert "language" in classify_topics("Rewrite the service in Go")
    assert "language" in classify_topics("Use Golang for the workers")


def test_supabase_classified():
    topics = classify_topics("Use Supabase for the backend")
    assert topics, "Supabase must classify (vocabulary gap in audit)"


def test_clerk_and_auth0_share_a_bucket():
    t1 = classify_topics("Use Clerk for authentication")
    t2 = classify_topics("Let's go with Auth0")
    assert t1 & t2, f"Clerk {t1} and Auth0 {t2} must share a topic bucket"


def test_multi_topic_statement_returns_all_topics():
    topics = classify_topics("Use FastAPI with SQLAlchemy and PostgreSQL for the backend")
    assert "web_framework" in topics
    assert "orm" in topics
    assert "database" in topics


def test_bigram_consumes_words_no_false_single_word_topic():
    """'session store' is a cache_backend bigram; the single word 'session'
    must NOT additionally contribute auth_mechanism."""
    topics = classify_topics("Use Redis as the session store")
    assert "cache_backend" in topics
    assert "auth_mechanism" not in topics


def test_unknown_tech_returns_empty_set():
    assert classify_topics("Adopt the flurbo protocol for widget sync") == set()


def test_classify_topic_backward_compat():
    """Singular wrapper still returns a str or None."""
    assert classify_topic("Use PostgreSQL for storage") == "database"
    assert classify_topic("Adopt the flurbo protocol") is None


# ── find_contradicting_decisions: set-intersection semantics ─────────────────

def _graph_with_decision(tmp_path, label, summary=""):
    g = GraphMemory(session_id="t-classifier", storage_dir=str(tmp_path))
    node_id = g.add_node(NodeType.DECISION, label, status=NodeStatus.COMPLETED,
                         summary=summary)
    return g, node_id


def test_db_switch_supersedes_multi_topic_decision(tmp_path):
    """THE audit scenario: multi-topic decision must be superseded when any
    of its topics is contradicted."""
    g, old_id = _graph_with_decision(
        tmp_path, "Use FastAPI with SQLAlchemy and PostgreSQL for the backend")
    hits = find_contradicting_decisions(
        "Switch from PostgreSQL to SQLite", "", g._nodes)
    assert old_id in hits


def test_unrelated_topic_does_not_supersede(tmp_path):
    g, old_id = _graph_with_decision(tmp_path, "Use PostgreSQL for storage")
    hits = find_contradicting_decisions("Use pytest as the test runner", "", g._nodes)
    assert old_id not in hits


def test_trpc_vs_grpc_supersedes(tmp_path):
    """Before the fix, 'Go with tRPC' classified as language, so a later
    gRPC decision never superseded it."""
    g, old_id = _graph_with_decision(tmp_path, "Go with tRPC for the API layer")
    hits = find_contradicting_decisions("Use gRPC instead for the API", "", g._nodes)
    assert old_id in hits


def test_supabase_to_firebase_supersedes(tmp_path):
    g, old_id = _graph_with_decision(tmp_path, "Use Supabase for the backend")
    hits = find_contradicting_decisions("Switch to Firebase", "", g._nodes)
    assert old_id in hits


def test_same_decision_not_superseded(tmp_path):
    g, old_id = _graph_with_decision(tmp_path, "Use PostgreSQL for storage")
    hits = find_contradicting_decisions("Use PostgreSQL for storage", "", g._nodes)
    assert old_id not in hits


# ── Near-duplicate decision merging ──────────────────────────────────────────

def test_containment_variant_merges_not_supersedes(tmp_path):
    """Two label variants of one decision (emitted from a single message)
    must merge into one node rather than supersede each other, which would
    record a spurious decision change."""
    g = GraphMemory(session_id="t-dup", storage_dir=str(tmp_path))
    id1 = g.add_node(NodeType.DECISION, "use React for the frontend.",
                     NodeStatus.COMPLETED)
    id2 = g.add_node(NodeType.DECISION, "Use React", NodeStatus.COMPLETED)
    assert id1 == id2, "containment variant must merge into the existing node"
    decisions = [n for n in g._nodes.values() if n.type == NodeType.DECISION]
    assert len(decisions) == 1
    assert decisions[0].status == NodeStatus.COMPLETED
    assert g.get_transitions() == [], "no self-supersession transition"


def test_longer_variant_upgrades_label(tmp_path):
    """When the more specific wording arrives second, keep it."""
    g = GraphMemory(session_id="t-dup2", storage_dir=str(tmp_path))
    id1 = g.add_node(NodeType.DECISION, "Use React", NodeStatus.COMPLETED)
    id2 = g.add_node(NodeType.DECISION, "Use React for the frontend",
                     NodeStatus.COMPLETED)
    assert id1 == id2
    assert g._nodes[id1].label == "Use React for the frontend"


def test_merge_does_not_resurrect_superseded(tmp_path):
    """Re-adding a same-decision variant as COMPLETED must not flip a
    SUPERSEDED node back to active."""
    g = GraphMemory(session_id="t-dup3", storage_dir=str(tmp_path))
    old_id = g.add_node(NodeType.DECISION, "Use React for the frontend",
                        NodeStatus.COMPLETED)
    g.add_node(NodeType.DECISION, "Switch to Next.js for the frontend",
               NodeStatus.COMPLETED)
    assert g._nodes[old_id].status == NodeStatus.SUPERSEDED
    again = g.add_node(NodeType.DECISION, "Use React", NodeStatus.COMPLETED)
    assert again == old_id
    assert g._nodes[old_id].status == NodeStatus.SUPERSEDED


def test_containment_needs_two_words():
    """{postgresql} alone must NOT collapse into 'Switch from PostgreSQL
    to SQLite' — single-word subsets match too many different decisions."""
    from tokenmizer.graph_memory.decision_tracker import _is_same_decision
    assert not _is_same_decision("PostgreSQL", "Switch from PostgreSQL to SQLite")
    assert not _is_same_decision("Use PostgreSQL", "Switch from PostgreSQL to SQLite")
    assert _is_same_decision("Use React", "use React for the frontend.")
    assert not _is_same_decision("Use PostgreSQL", "Do not use PostgreSQL",)
    assert _is_same_decision("Do not use Redis", "do not use Redis for caching",)


def test_semantic_opposites_are_not_same_decision():
    from tokenmizer.graph_memory.decision_tracker import _is_same_decision

    assert not _is_same_decision(
        "Enable JWT auth for the API",
        "Disable JWT auth for the API",
    )
    assert not _is_same_decision("Allow external signups", "Block external signups")
    assert not _is_same_decision("Use Redis", "Avoid Redis")
    assert _is_same_decision("Disable caching", "disable caching for the API")


def test_semantic_opposite_decisions_are_preserved(tmp_path):
    g = GraphMemory(session_id="t-semantic-opposites", storage_dir=str(tmp_path))

    enabled_id = g.add_node(
        NodeType.DECISION,
        "Enable JWT auth for the API",
        NodeStatus.COMPLETED,
    )
    disabled_id = g.add_node(
        NodeType.DECISION,
        "Disable JWT auth for the API",
        NodeStatus.COMPLETED,
    )

    assert disabled_id != enabled_id
    assert g._nodes[enabled_id].status == NodeStatus.SUPERSEDED
    assert g._nodes[disabled_id].status == NodeStatus.COMPLETED


def test_real_supersession_still_fires_after_merge_fix(tmp_path):
    """The merge fix must not swallow genuine contradictions."""
    g = GraphMemory(session_id="t-dup4", storage_dir=str(tmp_path))
    old_id = g.add_node(NodeType.DECISION, "Use React for the frontend",
                        NodeStatus.COMPLETED)
    new_id = g.add_node(NodeType.DECISION, "Switch to Next.js for the frontend",
                        NodeStatus.COMPLETED)
    assert new_id != old_id
    assert g._nodes[old_id].status == NodeStatus.SUPERSEDED
    assert len(g.get_transitions()) == 1


# ── Semantic slot-matching (Phase 2 of the memory-improvement plan) ─────────
#
# _same_slot / _find_by_word_overlap widen their lexical "is this a genuine
# replacement" check with embedding similarity via _semantic_same_slot
# (decision_tracker.py), reusing the same EmbeddingEngine reasoning.py's
# semantic recall already uses. Every test in this suite runs with
# EmbeddingEngine._load stubbed to a no-op (conftest.py), so .available is
# False and all the tests above this point behave IDENTICALLY without a
# model — that continues to hold, and is exactly why they didn't need any
# changes to keep passing. The "does it actually widen detection" behavior
# is tested here with a fake model (deterministic vectors, no network),
# matching the pattern already used in test_cache.py and test_reasoning.py.

def _install_fake_embedder(monkeypatch, vectors: dict):
    """Point the shared EmbeddingEngine singleton at fixed vectors for one
    test, with a pure-Python `.cosine` so no numpy is needed. monkeypatch
    restores the real (untrained, unavailable) state afterward."""
    from tokenmizer.semantic_cache.cache import EmbeddingEngine

    class _FakeModel:
        def encode(self, text_or_texts, normalize_embeddings=True):
            if isinstance(text_or_texts, str):
                return vectors[text_or_texts]
            return [vectors[t] for t in text_or_texts]

    def _dot(a, b) -> float:
        if a is None or b is None:
            return 0.0
        return float(sum(x * y for x, y in zip(a, b)))

    engine = EmbeddingEngine.get()
    monkeypatch.setattr(engine, "_model", _FakeModel())
    monkeypatch.setattr(EmbeddingEngine, "cosine", staticmethod(_dot))
    return engine


class TestSemanticSlotMatching:

    def test_paraphrased_same_purpose_now_supersedes(self, tmp_path, monkeypatch):
        """'primary datastore' vs 'main persistence layer' share zero
        slot words — before Phase 2 this stayed CONTESTED forever."""
        _install_fake_embedder(monkeypatch, {
            "Use PostgreSQL for the primary datastore":
                [1.0, 0.0],
            "Switch to CockroachDB as the main persistence layer":
                [0.95, 0.312],   # cos = 0.95 — clearly the same purpose
        })
        g = GraphMemory(session_id="t-sem-slot-1", storage_dir=str(tmp_path))
        old_id = g.add_node(NodeType.DECISION,
                            "Use PostgreSQL for the primary datastore",
                            NodeStatus.COMPLETED)
        new_id = g.add_node(NodeType.DECISION,
                            "Switch to CockroachDB as the main persistence layer",
                            NodeStatus.COMPLETED)
        assert new_id != old_id
        assert g._nodes[old_id].status == NodeStatus.SUPERSEDED, (
            f"expected SUPERSEDED via semantic slot match, got {g._nodes[old_id].status}"
        )

    def test_genuinely_complementary_decisions_stay_contested(self, tmp_path, monkeypatch):
        """The exact TM-09 case (test_contested_decisions.py) must not
        regress: same topic, genuinely different purpose. A real model
        would score this pair well below the 0.72 threshold; simulated
        here with an explicitly low fake similarity to prove the
        threshold — not just "no model" — is what keeps it correctly
        unmerged."""
        _install_fake_embedder(monkeypatch, {
            "Use PostgreSQL for primary user data": [1.0, 0.0],
            "Use SQLite for the local offline cache": [0.3, 0.954],  # cos = 0.3
        })
        g = GraphMemory(session_id="t-sem-slot-2", storage_dir=str(tmp_path))
        old_id = g.add_node(NodeType.DECISION, "Use PostgreSQL for primary user data",
                            NodeStatus.COMPLETED, summary="relational integrity matters")
        new_id = g.add_node(NodeType.DECISION, "Use SQLite for the local offline cache",
                            NodeStatus.COMPLETED, summary="zero-config embedded")
        assert g._nodes[old_id].status == NodeStatus.CONTESTED
        assert g._nodes[new_id].status == NodeStatus.CONTESTED

    def test_unknown_topic_fallback_is_also_widened(self, tmp_path, monkeypatch):
        """_find_by_word_overlap (the unknown-topic path — see its
        docstring) gets the identical fix, not just _same_slot's
        known-topic path — same root cause, two call sites."""
        from tokenmizer.graph_memory.decision_tracker import find_contradicting_decisions

        _install_fake_embedder(monkeypatch, {
            "Adopt Zephyrmesh for service discovery": [1.0, 0.0],
            "Move to Nebulawire as our discovery layer": [0.9, 0.436],  # cos = 0.9
        })
        g = GraphMemory(session_id="t-sem-slot-3", storage_dir=str(tmp_path))
        old_id = g.add_node(NodeType.DECISION,
                            "Adopt Zephyrmesh for service discovery",
                            NodeStatus.COMPLETED)
        assert classify_topics("Adopt Zephyrmesh for service discovery") == set(), (
            "fixture assumes these made-up tech names hit zero taxonomy keywords"
        )
        hits = find_contradicting_decisions(
            "Move to Nebulawire as our discovery layer", "", g._nodes,
        )
        assert old_id in hits

    def test_no_semantic_widening_without_an_embedding_model(self, tmp_path):
        """Every other test in this file runs exactly this way (see the
        module comment above) — asserted explicitly so a future change
        to the default stub is caught."""
        from tokenmizer.semantic_cache.cache import EmbeddingEngine
        assert EmbeddingEngine.get().available is False
        g = GraphMemory(session_id="t-sem-slot-4", storage_dir=str(tmp_path))
        old_id = g.add_node(NodeType.DECISION,
                            "Use PostgreSQL for the primary datastore",
                            NodeStatus.COMPLETED)
        new_id = g.add_node(NodeType.DECISION,
                            "Switch to CockroachDB as the main persistence layer",
                            NodeStatus.COMPLETED)
        # Zero shared slot words AND no model available — stays contested,
        # exactly like every ambiguous case did before Phase 2.
        assert new_id != old_id
        assert g._nodes[old_id].status == NodeStatus.CONTESTED


# ── Semantic error dedup (Phase 3) ───────────────────────────────────────────
#
# _is_same_error / _semantic_same_error / _named_error_classes_conflict —
# see decision_tracker.py's "Error dedup" section. Reuses the same
# _install_fake_embedder helper and no-network/no-numpy approach as the
# semantic-slot tests above.

class TestSemanticErrorDedup:

    def test_paraphrased_same_failure_is_recognized(self, monkeypatch):
        from tokenmizer.graph_memory.decision_tracker import _semantic_same_error

        _install_fake_embedder(monkeypatch, {
            "Connection to the database times out": [1.0, 0.0],
            "DB connection timeout after 30s on checkout": [0.95, 0.312],  # cos = 0.95
        })
        assert _semantic_same_error(
            "DB connection timeout after 30s on checkout",
            "Connection to the database times out",
        ) is True

    def test_unrelated_errors_are_not_merged(self, monkeypatch):
        from tokenmizer.graph_memory.decision_tracker import _semantic_same_error

        _install_fake_embedder(monkeypatch, {
            "Connection to the database times out": [1.0, 0.0],
            "CSS layout breaks on mobile Safari": [0.3, 0.954],  # cos = 0.3
        })
        assert _semantic_same_error(
            "CSS layout breaks on mobile Safari",
            "Connection to the database times out",
        ) is False

    def test_named_exception_class_conflict_overrides_high_similarity(self, monkeypatch):
        """Even a fake similarity score of 1.0 must not merge two labels
        naming different exception classes — the guard is checked FIRST
        and short-circuits before the embedding comparison runs at all."""
        from tokenmizer.graph_memory.decision_tracker import _semantic_same_error

        _install_fake_embedder(monkeypatch, {
            "TypeError: x is not a function": [1.0, 0.0],
            "ReferenceError: x is not a function": [1.0, 0.0],  # cos = 1.0
        })
        assert _semantic_same_error(
            "ReferenceError: x is not a function",
            "TypeError: x is not a function",
        ) is False

    def test_no_semantic_merge_without_an_embedding_model(self):
        from tokenmizer.graph_memory.decision_tracker import _semantic_same_error
        from tokenmizer.semantic_cache.cache import EmbeddingEngine

        assert EmbeddingEngine.get().available is False
        assert _semantic_same_error(
            "DB connection timeout after 30s on checkout",
            "Connection to the database times out",
        ) is False
