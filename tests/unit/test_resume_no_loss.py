"""Resume blocks must not drop, invent, or repeat facts.

Each case below is a defect found by reading the resume text the
checkpoint_accuracy sessions actually produce, not just their scores.
"""
from __future__ import annotations

from tokenmizer.checkpoints.manager import CheckpointManager
from tokenmizer.graph_memory.context_block import drop_restated
from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType


def _graph(tmp_path, reply: str, user: str = "go") -> GraphMemory:
    g = GraphMemory("t", storage_dir=str(tmp_path))
    g.extract_from_messages(
        [{"role": "user", "content": user}, {"role": "assistant", "content": reply}],
        incremental=False,
    )
    return g


def _errors(g: GraphMemory) -> dict[str, NodeStatus]:
    return {n.label: n.status for n in g._nodes.values() if n.type == NodeType.ERROR}


# ── Errors: a fixed bug's cause is not an open bug ─────────────────────────

def test_cause_clause_of_a_fix_is_resolved(tmp_path):
    g = _graph(tmp_path, "Fixed: WebSocket message not triggering re-render — was "
                         "missing dependency in useEffect. Chart now updates live.")
    errs = _errors(g)
    assert "missing dependency in useEffect" in errs
    assert all(s != NodeStatus.FAILED for s in errs.values()), errs


def test_but_still_keeps_the_remaining_bug_open(tmp_path):
    g = _graph(tmp_path, "Fixed: the login 422, but still missing CSRF token in the logout form.")
    errs = _errors(g)
    assert errs.get("missing CSRF token in the logout form") == NodeStatus.FAILED, errs


def test_absence_label_keeps_its_place(tmp_path):
    g = _graph(tmp_path, "The build is missing a lockfile in CI. Deploys fail.")
    assert "missing a lockfile in CI" in _errors(g)


# ── Done: different tasks with a shared prefix all stay ────────────────────

def test_done_keeps_tasks_with_a_common_prefix(tmp_path):
    g = GraphMemory("t", storage_dir=str(tmp_path))
    for ep in ("register", "login", "logout"):
        g.add_node(NodeType.TASK, f"Implemented POST /api/auth/{ep}", NodeStatus.COMPLETED)
    block = g.to_context_block(token_budget=400)
    for ep in ("register", "login", "logout"):
        assert f"/api/auth/{ep}" in block, block


# ── Decisions: restatements go, nothing else does ──────────────────────────

def test_drop_restated_removes_only_fully_covered_labels():
    labels = [
        "Use JWT and PostgreSQL",
        "Python 3.12 and bcrypt for password hashing",
        "Python 3.12 runtime",
        "bcrypt for password hashing (industry standard)",
        "Redis for refresh token storage",
    ]
    assert drop_restated(labels) == [
        "Use JWT and PostgreSQL",
        "Python 3.12 runtime",
        "bcrypt for password hashing (industry standard)",
        "Redis for refresh token storage",
    ]


def test_drop_restated_does_not_pair_facts_from_different_labels():
    # Every word of the third appears somewhere, but no single label says it.
    labels = ["Postgres for users", "Redis for sessions", "Redis for users"]
    assert drop_restated(labels) == labels


def test_drop_restated_keeps_negations():
    labels = ["JWT in httpOnly cookie", "JWT not in localStorage", "localStorage for theme"]
    assert drop_restated(labels) == labels


def test_drop_restated_keeps_first_of_identical_labels():
    assert drop_restated(["Use Redis", "Use Redis"]) == ["Use Redis"]


def test_critical_tier_does_not_spend_slots_on_restatements(tmp_path):
    g = GraphMemory("t", storage_dir=str(tmp_path))
    for label in ("Python 3.12 and bcrypt for password hashing", "Python 3.12 runtime",
                  "bcrypt for password hashing (industry standard)",
                  "Redis for refresh token storage"):
        g.add_node(NodeType.DECISION, label, NodeStatus.COMPLETED)
    ckpt = CheckpointManager(storage_dir=str(tmp_path)).create(
        session_id="t", messages=[{"role": "user", "content": "go"}], graph=g, context_pct=0.9)
    assert "Redis for refresh token storage" in ckpt.resume_critical
    assert "Python 3.12 and bcrypt" not in ckpt.resume_critical


# ── Full tier: no line repeats what the standard block already says ───────

def test_full_tier_does_not_repeat_done_or_env(tmp_path):
    g = GraphMemory("t", storage_dir=str(tmp_path))
    g.add_node(NodeType.TASK, "User model in api/models.py", NodeStatus.COMPLETED)
    g.add_node(NodeType.ENVIRONMENT, "Python 3.12", NodeStatus.COMPLETED)
    ckpt = CheckpointManager(storage_dir=str(tmp_path)).create(
        session_id="t", messages=[{"role": "user", "content": "go"}], graph=g, context_pct=0.9)
    full = ckpt.resume_full
    assert full.count("User model in api/models.py") == 1, full
    assert full.count("Python 3.12") == 1, full
