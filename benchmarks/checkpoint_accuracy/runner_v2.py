"""
Benchmark v2 — Three-way comparison:
  Heuristic Only  vs  Hybrid (Heuristic+LLM)  vs  Summary (no graph)

This is the proof layer. Run this to generate the comparison table
that answers: "Does graph memory actually beat a plain summary?"

Run: python -m benchmarks.checkpoint_accuracy.runner_v2
"""
from __future__ import annotations

import json
import re
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

# ── Benchmark sessions (diverse domains) ─────────────────────────────────────

SESSIONS = {
    "fastapi_auth": {
        "messages": [
            {"role": "user",      "content": "Let's build a FastAPI authentication service with JWT and PostgreSQL"},
            {"role": "assistant", "content": "Starting project. Creating: api/main.py, api/auth.py, api/models.py, config.py, requirements.txt"},
            {"role": "user",      "content": "Use Python 3.12 and bcrypt for password hashing"},
            {"role": "assistant", "content": "Decided: Python 3.12 runtime. Decided: bcrypt for password hashing (industry standard). Completed: virtual environment setup."},
            {"role": "user",      "content": "Implement the user model first"},
            {"role": "assistant", "content": "Completed: User model in api/models.py — fields: id, email, hashed_password, created_at, is_active"},
            {"role": "user",      "content": "Now add the auth endpoints"},
            {"role": "assistant", "content": "Implemented: POST /api/auth/register, POST /api/auth/login (returns JWT), POST /api/auth/logout. Files updated: api/auth.py"},
            {"role": "user",      "content": "Login keeps returning 422"},
            {"role": "assistant", "content": "Fixed: 422 error — missing email validation in LoginRequest Pydantic model. Updated api/models.py. Login endpoint working now."},
            {"role": "user",      "content": "Add refresh token support using Redis"},
            {"role": "assistant", "content": "Decided: Redis for refresh token storage (not DB — faster revocation). Adding: redis to requirements.txt. Working on: refresh token rotation in api/auth.py"},
            {"role": "user",      "content": "What's the DB schema look like?"},
            {"role": "assistant", "content": "Schema: users table — id (UUID PK), email (unique), hashed_password, created_at, is_active. No refresh_tokens table — using Redis."},
            {"role": "user",      "content": "Write tests"},
            {"role": "assistant", "content": "Created tests/test_auth.py with 12 tests: login_success, login_wrong_password, register_success, token_refresh, token_expiry, logout. All 12 passing."},
            {"role": "user",      "content": "Add rate limiting to auth endpoints"},
            {"role": "assistant", "content": "Working on: rate limiting using slowapi. Will limit: 5 login attempts per minute per IP. Updating api/main.py"},
        ],
        "ground_truth": {
            "completed_tasks": ["project structure", "user model", "auth endpoints", "fix 422", "tests"],
            "pending_tasks":   ["refresh token", "rate limiting"],
            "decisions":       ["postgresql", "python 3.12", "bcrypt", "redis", "jwt"],
            "files":           ["api/auth.py", "api/models.py", "api/main.py", "config.py", "tests/test_auth.py"],
        },
    },

    "react_dashboard": {
        "messages": [
            {"role": "user",      "content": "Building a real-time analytics dashboard with React and WebSockets"},
            {"role": "assistant", "content": "Starting dashboard project. Creating: src/App.tsx, src/components/Dashboard.tsx, src/hooks/useWebSocket.ts"},
            {"role": "user",      "content": "Use TypeScript and Recharts for charts"},
            {"role": "assistant", "content": "Decided: TypeScript for type safety. Decided: Recharts for data visualization (good React integration). Completed: project scaffold with Vite."},
            {"role": "user",      "content": "The chart is not updating in real time"},
            {"role": "assistant", "content": "Fixed: WebSocket message not triggering re-render — was missing dependency in useEffect. Updated src/hooks/useWebSocket.ts. Chart now updates live."},
            {"role": "user",      "content": "Add dark mode"},
            {"role": "assistant", "content": "Decided: CSS variables for theming (no extra library). Completed: dark mode toggle in src/components/Header.tsx. Updated src/styles/theme.css."},
            {"role": "user",      "content": "Add authentication"},
            {"role": "assistant", "content": "Going with JWT stored in httpOnly cookie (not localStorage — XSS safe). Working on: src/hooks/useAuth.ts, src/components/Login.tsx"},
        ],
        "ground_truth": {
            "completed_tasks": ["project scaffold", "chart realtime fix", "dark mode"],
            "pending_tasks":   ["authentication"],
            "decisions":       ["typescript", "recharts", "css variables", "jwt", "httponly cookie"],
            "files":           ["src/App.tsx", "src/hooks/useWebSocket.ts", "src/hooks/useAuth.ts", "src/components/Dashboard.tsx"],
        },
    },

    "ml_pipeline": {
        "messages": [
            {"role": "user",      "content": "Need to build a training pipeline for a text classification model"},
            {"role": "assistant", "content": "Building ML pipeline. Creating: train.py, data/preprocess.py, models/classifier.py, config.yaml"},
            {"role": "user",      "content": "Use PyTorch and HuggingFace transformers"},
            {"role": "assistant", "content": "Decided: PyTorch for training framework. Decided: HuggingFace transformers, using bert-base-uncased. Completed: environment setup with Python 3.11."},
            {"role": "user",      "content": "Training is too slow"},
            {"role": "assistant", "content": "Fixed: Added gradient checkpointing and mixed precision (torch.cuda.amp). Training 3x faster. Updated train.py."},
            {"role": "user",      "content": "Switch from BERT to DistilBERT — faster inference"},
            {"role": "assistant", "content": "Switching from bert-base-uncased to distilbert-base-uncased. Reason: 40% faster inference, only 3% accuracy drop for our task. Updated models/classifier.py and config.yaml."},
            {"role": "user",      "content": "Add evaluation metrics"},
            {"role": "assistant", "content": "Completed: evaluation script in evaluate.py with precision, recall, F1, and confusion matrix. Using sklearn.metrics."},
        ],
        "ground_truth": {
            "completed_tasks": ["environment setup", "gradient checkpointing fix", "evaluation metrics"],
            "pending_tasks":   [],
            "decisions":       ["pytorch", "huggingface", "distilbert", "mixed precision"],
            "files":           ["train.py", "models/classifier.py", "config.yaml", "evaluate.py"],
        },
    },
}


# ── Scoring ───────────────────────────────────────────────────────────────────

def _fuzzy_match(a: str, b: str) -> bool:
    a, b = a.lower().strip(), b.lower().strip()
    if a in b or b in a:
        return True
    wa = set(re.findall(r'\w{3,}', a))
    wb = set(re.findall(r'\w{3,}', b))
    if not wa or not wb:
        return False
    shorter = wa if len(wa) <= len(wb) else wb
    return len(wa & wb) / len(shorter) >= 0.50


def _recall(extracted: set[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    if not extracted:
        return 0.0
    matched = sum(1 for e in expected if any(_fuzzy_match(e, ex) for ex in extracted))
    return round(matched / len(expected), 3)


def _precision(extracted: set[str], expected: list[str]) -> float:
    if not extracted:
        return 1.0
    matched = sum(1 for ex in extracted if any(_fuzzy_match(ex, e) for e in expected))
    return round(matched / len(extracted), 3)


# ── Summary baseline (naive approach — no graph) ──────────────────────────────

def _summary_extract(messages: list[dict]) -> dict:
    """
    Naive summary: just grab last N messages and count what's there.
    Represents what you'd get from a plain 'summarize this conversation' approach.
    No graph, no structured extraction.
    """
    # Plain text extraction — simple keyword scan, no structure
    all_text = " ".join(m.get("content","") for m in messages)
    decisions, tasks, files = set(), set(), set()

    for word in ["postgresql","redis","jwt","bcrypt","python","typescript","pytorch","distilbert","recharts"]:
        if word in all_text.lower():
            decisions.add(word)

    for pattern in [r'\b\w+\.py\b', r'\bsrc/\S+\b', r'\btests/\S+\b']:
        for m in re.finditer(pattern, all_text):
            files.add(m.group(0).strip(",."))

    for kw in ["completed:", "fixed:", "implemented:", "created:"]:
        idx = 0
        while True:
            idx = all_text.lower().find(kw, idx)
            if idx == -1:
                break
            snippet = all_text[idx+len(kw):idx+len(kw)+50].split(".")[0].strip()
            if snippet:
                tasks.add(snippet)
            idx += 1

    return {"decisions": decisions, "tasks": tasks, "files": files}


# ── Runner ────────────────────────────────────────────────────────────────────

@dataclass
class ComparisonResult:
    session: str
    turns: int
    # Heuristic
    heu_task_recall: float
    heu_decision_recall: float
    heu_file_recall: float
    heu_time_ms: float
    heu_nodes: int
    # Summary (naive baseline)
    sum_task_recall: float
    sum_decision_recall: float
    sum_file_recall: float
    # Resume
    resume_tokens: int

    def info_preserved_heuristic(self) -> float:
        return round((self.heu_task_recall + self.heu_decision_recall + self.heu_file_recall) / 3, 3)

    def info_preserved_summary(self) -> float:
        return round((self.sum_task_recall + self.sum_decision_recall + self.sum_file_recall) / 3, 3)

    def graph_vs_summary_delta(self) -> float:
        return round(self.info_preserved_heuristic() - self.info_preserved_summary(), 3)


def run_comparison(name: str, session: dict) -> ComparisonResult:
    messages = session["messages"]
    gt = session["ground_truth"]

    # ── Heuristic graph extraction ────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        g = GraphMemory(f"bench-{name}", storage_dir=tmp)
        mgr = CheckpointManager(storage_dir=tmp)

        t0 = time.monotonic()
        g.extract_from_messages(messages, incremental=False)
        heu_ms = (time.monotonic() - t0) * 1000

        ckpt = mgr.create(
            session_id=f"bench-{name}",
            messages=messages, graph=g, context_pct=0.9,
        )

    task_labels    = {n.label for n in g._nodes.values() if n.type == NodeType.TASK and n.status == NodeStatus.COMPLETED}
    decision_labels = {n.label for n in g._nodes.values() if n.type == NodeType.DECISION}
    file_labels    = {n.label for n in g._nodes.values() if n.type == NodeType.FILE}

    heu_tr = _recall(task_labels, gt["completed_tasks"])
    heu_dr = _recall(decision_labels, gt["decisions"])
    heu_fr = _recall(file_labels, gt["files"])

    # ── Summary baseline ──────────────────────────────────────────────────
    summary = _summary_extract(messages)
    sum_tr = _recall(summary["tasks"], gt["completed_tasks"])
    sum_dr = _recall(summary["decisions"], gt["decisions"])
    sum_fr = _recall(summary["files"], gt["files"])

    return ComparisonResult(
        session=name, turns=len(messages),
        heu_task_recall=heu_tr, heu_decision_recall=heu_dr, heu_file_recall=heu_fr,
        heu_time_ms=heu_ms, heu_nodes=len(g._nodes),
        sum_task_recall=sum_tr, sum_decision_recall=sum_dr, sum_file_recall=sum_fr,
        resume_tokens=ckpt.resume_tokens,
    )


def run_all(save_json: bool = True) -> list[ComparisonResult]:
    print("\n🧠 TokenMizer — Benchmark v2: Graph vs Summary")
    print("=" * 65)
    print(f"{'Session':<22} {'Method':<12} {'Task R':>7} {'Decision R':>11} {'File R':>7} {'Info%':>7}")
    print("-" * 65)

    results = []
    for name, session in SESSIONS.items():
        r = run_comparison(name, session)
        results.append(r)

        print(f"{name:<22} {'Graph':<12} {r.heu_task_recall:>7.0%} {r.heu_decision_recall:>11.0%} {r.heu_file_recall:>7.0%} {r.info_preserved_heuristic():>7.0%}")
        print(f"{'':22} {'Summary':<12} {r.sum_task_recall:>7.0%} {r.sum_decision_recall:>11.0%} {r.sum_file_recall:>7.0%} {r.info_preserved_summary():>7.0%}")
        delta = r.graph_vs_summary_delta()
        sign = "+" if delta >= 0 else ""
        print(f"{'':22} {'Δ Graph-Sum':<12} {'':>7} {'':>11} {'':>7} {sign}{delta:>6.0%}")
        print(f"{'':22} {'(tokens)':<12} resume={r.resume_tokens}t  nodes={r.heu_nodes}  {r.heu_time_ms:.0f}ms")
        print()

    # Averages
    n = len(results)
    avg_heu = sum(r.info_preserved_heuristic() for r in results) / n
    avg_sum = sum(r.info_preserved_summary() for r in results) / n
    avg_delta = avg_heu - avg_sum
    sign = "+" if avg_delta >= 0 else ""

    print("=" * 65)
    print(f"{'AVERAGE':<22} {'Graph':<12} {'':>7} {'':>11} {'':>7} {avg_heu:>7.0%}")
    print(f"{'':22} {'Summary':<12} {'':>7} {'':>11} {'':>7} {avg_sum:>7.0%}")
    print(f"{'':22} {'Δ advantage':<12} {'':>7} {'':>11} {'':>7} {sign}{avg_delta:>6.0%}")
    print()

    avg_task_r = sum(r.heu_task_recall for r in results) / n
    avg_dec_r  = sum(r.heu_decision_recall for r in results) / n
    avg_file_r = sum(r.heu_file_recall for r in results) / n
    avg_tokens = sum(r.resume_tokens for r in results) / n

    print("Detailed averages (Graph/Heuristic only):")
    print(f"  Task Recall:      {avg_task_r:.0%}")
    print(f"  Decision Recall:  {avg_dec_r:.0%}")
    print(f"  File Recall:      {avg_file_r:.0%}")
    print(f"  Avg resume size:  {avg_tokens:.0f} tokens")
    print()

    if save_json:
        out = Path("benchmark_v2_results.json")
        out.write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"Saved: {out}")

    return results


if __name__ == "__main__":
    run_all()
