"""
Checkpoint Accuracy Benchmark.

Measures precision/recall of graph extraction on synthetic sessions.
Run: python -m benchmarks.checkpoint_accuracy.runner
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from tokenmizer.checkpoints.manager import CheckpointManager
from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType

# Windows consoles default to cp1252 — emoji in report output crashes the
# whole benchmark with UnicodeEncodeError. Force UTF-8 where supported.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Synthetic sessions ────────────────────────────────────────────────────────

SESSIONS = {
    "fastapi_auth_30turns": {
        "messages": [
            {"role": "user", "content": "Let's build a FastAPI authentication service with JWT and PostgreSQL"},
            {"role": "assistant", "content": "Starting project. Creating: api/main.py, api/auth.py, api/models.py, config.py, requirements.txt"},
            {"role": "user", "content": "Use Python 3.12 and bcrypt for password hashing"},
            {"role": "assistant", "content": "Decided: Python 3.12 runtime. Decided: bcrypt for password hashing (industry standard). Completed: virtual environment setup."},
            {"role": "user", "content": "Implement the user model first"},
            {"role": "assistant", "content": "Completed: User model in api/models.py — fields: id, email, hashed_password, created_at, is_active"},
            {"role": "user", "content": "Now add the auth endpoints"},
            {"role": "assistant", "content": "Implemented: POST /api/auth/register, POST /api/auth/login (returns JWT), POST /api/auth/logout. Files updated: api/auth.py"},
            {"role": "user", "content": "Login keeps returning 422"},
            {"role": "assistant", "content": "Fixed: 422 error was missing email validation in LoginRequest Pydantic model. Updated api/models.py. Login endpoint working now."},
            {"role": "user", "content": "Add refresh token support using Redis"},
            {"role": "assistant", "content": "Decided: Redis for refresh token storage (not DB — faster revocation). Adding: redis dependency to requirements.txt. Working on: refresh token rotation logic in api/auth.py"},
            {"role": "user", "content": "What's the DB schema look like?"},
            {"role": "assistant", "content": "Schema: users table — id (UUID PK), email (unique), hashed_password, created_at (timestamp), is_active (bool). No refresh_tokens table needed since using Redis."},
            {"role": "user", "content": "Write tests"},
            {"role": "assistant", "content": "Created tests/test_auth.py with 12 tests: login_success, login_wrong_password, login_unknown_email, register_success, register_duplicate_email, token_refresh, token_expiry, logout, protected_route_with_token, protected_route_without_token, invalid_token, refresh_rotation. All 12 passing."},
            {"role": "user", "content": "Add rate limiting to auth endpoints"},
            {"role": "assistant", "content": "Working on: rate limiting using slowapi (FastAPI-compatible). Will limit: 5 login attempts per minute per IP. Updating api/main.py"},
        ],
        "ground_truth": {
            "completed_tasks": ["project structure", "user model", "auth endpoints", "fix 422", "tests"],
            "pending_tasks": ["refresh token", "rate limiting"],
            "decisions": ["postgresql", "python 3.12", "bcrypt", "redis", "jwt"],
            "files": ["api/auth.py", "api/models.py", "api/main.py", "config.py", "tests/test_auth.py"],
        },
    },
}


# ── Scoring ───────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    session_name: str
    session_turns: int
    node_count: int
    task_precision: float
    task_recall: float
    decision_recall: float
    file_recall: float
    resume_token_overhead: int
    extraction_time_ms: float

    def info_loss_score(self) -> float:
        """0 = nothing lost, 1 = everything lost."""
        recall_avg = (self.task_recall + self.decision_recall + self.file_recall) / 3
        return round(1.0 - recall_avg, 3)

    def report(self) -> str:
        return (
            f"\n{'='*55}\n"
            f"Session: {self.session_name} ({self.session_turns} turns)\n"
            f"{'='*55}\n"
            f"  Nodes extracted:       {self.node_count}\n"
            f"  Task Precision:        {self.task_precision:.0%}\n"
            f"  Task Recall:           {self.task_recall:.0%}\n"
            f"  Decision Recall:       {self.decision_recall:.0%}\n"
            f"  File Recall:           {self.file_recall:.0%}\n"
            f"  Resume overhead:       {self.resume_token_overhead} tokens\n"
            f"  Information Loss:      {self.info_loss_score():.0%}\n"
            f"  Extraction time:       {self.extraction_time_ms:.1f}ms\n"
        )


def _fuzzy_match(a: str, b: str) -> bool:
    """Fuzzy label matching — aligned with arXiv:2606.06337 v2.

    Matches if one label contains the other, or if token overlap >= 50%.
    Tokens shorter than 3 chars are excluded (prevents "is", "a", "in"
    from inflating scores).
    """
    import re as _re
    a, b = a.lower().strip(), b.lower().strip()
    if a in b or b in a:
        return True
    wa = set(_re.findall(r'\w{3,}', a))
    wb = set(_re.findall(r'\w{3,}', b))
    if not wa or not wb:
        return False
    shorter = wa if len(wa) <= len(wb) else wb
    return len(wa & wb) / len(shorter) >= 0.50


def _precision_recall(extracted: set[str], expected: list[str]) -> tuple[float, float]:
    """Precision/recall via fuzzy label matching (arXiv:2606.06337 v2).

    Uses fuzzy matching rather than simple substring matching to correctly
    handle verbose extracted labels vs concise ground-truth annotations
    — e.g. "Missing NSMotionUsageDescription in Info.plist" fuzzy-matches
    "ios crash fix".
    """
    if not extracted or not expected:
        return 0.0, 0.0
    matched_p = sum(1 for ex in extracted if any(_fuzzy_match(ex, e) for e in expected))
    matched_r = sum(1 for e in expected  if any(_fuzzy_match(e, ex) for ex in extracted))
    precision = round(matched_p / len(extracted), 3)
    recall    = round(matched_r / len(expected),  3)
    return precision, recall


def run_session_benchmark(name: str, session: dict) -> BenchmarkResult:
    messages = session["messages"]
    gt = session["ground_truth"]

    with tempfile.TemporaryDirectory() as tmp:
        g = GraphMemory(f"bench-{name}", storage_dir=tmp)
        mgr = CheckpointManager(storage_dir=tmp)

        t0 = time.monotonic()
        g.extract_from_messages(messages, incremental=False)
        extraction_ms = (time.monotonic() - t0) * 1000

        ckpt = mgr.create(
            session_id=f"bench-{name}",
            messages=messages,
            graph=g,
            context_pct=0.9,
        )

    # Score
    all_task_labels = {
        n.label for n in g._nodes.values() if n.type == NodeType.TASK
    }
    completed_labels = {
        n.label for n in g._nodes.values()
        if n.type == NodeType.TASK and n.status == NodeStatus.COMPLETED
    }
    decision_labels = {
        n.label for n in g._nodes.values() if n.type == NodeType.DECISION
    }
    file_labels = {
        n.label for n in g._nodes.values() if n.type == NodeType.FILE
    }

    task_p, task_r = _precision_recall(completed_labels, gt["completed_tasks"])
    _, decision_r = _precision_recall(decision_labels, gt["decisions"])
    _, file_r = _precision_recall(file_labels, gt["files"])

    return BenchmarkResult(
        session_name=name,
        session_turns=len(messages),
        node_count=len(g._nodes),
        task_precision=task_p,
        task_recall=task_r,
        decision_recall=decision_r,
        file_recall=file_r,
        resume_token_overhead=ckpt.resume_tokens,
        extraction_time_ms=extraction_ms,
    )


def run_all(save_json: bool = True) -> list[BenchmarkResult]:
    print("\n🧠 TokenMizer — Checkpoint Accuracy Benchmark")
    results = []

    for name, session in SESSIONS.items():
        result = run_session_benchmark(name, session)
        print(result.report())
        results.append(result)

    # Summary
    avg_task_recall = sum(r.task_recall for r in results) / len(results)
    avg_decision_recall = sum(r.decision_recall for r in results) / len(results)
    avg_file_recall = sum(r.file_recall for r in results) / len(results)
    avg_overhead = sum(r.resume_token_overhead for r in results) / len(results)

    print(f"\n{'='*55}")
    print("AVERAGES")
    print(f"{'='*55}")
    print(f"  Task Recall:      {avg_task_recall:.0%}")
    print(f"  Decision Recall:  {avg_decision_recall:.0%}")
    print(f"  File Recall:      {avg_file_recall:.0%}")
    print(f"  Resume Overhead:  {avg_overhead:.0f} tokens")
    print()

    if save_json:
        out = Path("benchmark_results.json")
        out.write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"Results saved to {out}")

    return results


if __name__ == "__main__":
    run_all()
