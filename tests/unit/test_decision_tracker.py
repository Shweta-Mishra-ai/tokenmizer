"""
Regression tests for the topic classifier (2026-07-10 audit).

Confirmed misclassifications before the fix (all reproduced by execution):
  - "Go with tRPC for the API layer"  → "language" (imperative "Go" collided
    with Go-the-language; first-single-word-hit-wins never reached "trpc")
  - "Use Supabase for the backend"    → None (vocabulary gap; the sibling
    hybrid_extractor.py knew "supabase" but this file didn't)
  - "Use Clerk for authentication"    → None (same gap)
  - "Use FastAPI with SQLAlchemy and PostgreSQL" → only "web_framework";
    a later Postgres→SQLite switch was never detected as contradicting it.
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


# ── AUDIT round 2 (2026-07-10): near-duplicate decision merging ──────────────

def test_containment_variant_merges_not_supersedes(tmp_path):
    """The demo bug: 'use React for the frontend.' and 'Use React' were
    emitted from ONE message, became two nodes, and one superseded the
    other — a bogus 'Changed:' line in every resume. They must merge."""
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
