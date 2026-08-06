"""
Graph memory retrieval quality benchmark.
Tests recall of tasks, decisions, and files after synthetic sessions.

Run: python -m benchmarks.graph_retrieval.runner
No server required — tests graph module directly.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import tempfile

from tokenmizer.graph_memory.graph import GraphMemory

SYNTHETIC_SESSION = [
    {"role": "user", "content": "Let's build a FastAPI auth service with JWT and PostgreSQL."},
    {"role": "assistant", "content": "Great goal. I'll help you build the auth service."},
    {"role": "user", "content": "I've decided to use bcrypt for password hashing instead of argon2."},
    {"role": "assistant", "content": "Good choice. bcrypt is well-supported."},
    {"role": "user", "content": "Completed the user model in api/models.py. Tests passing."},
    {"role": "assistant", "content": "User model looks solid."},
    {"role": "user", "content": "Working on the login endpoint in api/auth.py now."},
    {"role": "assistant", "content": "Let me know when you hit any issues."},
    {"role": "user", "content": "Fixed the 422 validation error. Using Redis for refresh tokens."},
    {"role": "assistant", "content": "Redis is a solid choice for token storage."},
    {"role": "user", "content": "Switching from React to Next.js for better SEO."},
    {"role": "assistant", "content": "Makes sense given the SEO requirement."},
    {"role": "user", "content": "Environment: Python 3.12, FastAPI 0.111, PostgreSQL 16"},
    {"role": "assistant", "content": "Noted the environment details."},
]

GROUND_TRUTH = {
    "goals": ["FastAPI auth service with JWT and PostgreSQL"],
    "tasks_done": ["user model", "login endpoint", "fix 422"],
    "decisions": ["bcrypt", "Redis for refresh tokens"],
    "files": ["api/models.py", "api/auth.py"],
    "superseded": [("React", "Next.js")],
    "environments": ["Python 3.12", "FastAPI 0.111", "PostgreSQL 16"],
}


def recall(found: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    found_lower = " ".join(found).lower()
    hits = sum(1 for e in expected if e.lower() in found_lower)
    return hits / len(expected)


def run():
    print("\nGraph Memory Retrieval Benchmark")
    print("  Mode: Heuristic-only (set use_llm_extraction=true for LLM mode)")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        g = GraphMemory("graph-retrieval-benchmark", storage_dir=tmpdir)
        g.extract_from_messages(SYNTHETIC_SESSION)

        nodes = list(g._nodes.values())
        node_labels = [n.label for n in nodes]

        scores = {}
        for category, expected in GROUND_TRUTH.items():
            if category == "superseded":
                # Check if superseded decisions are tracked
                superseded = [n for n in nodes if n.status.value == "superseded"]
                hit = any("react" in n.label.lower() for n in superseded)
                scores[category] = 1.0 if hit else 0.0
            else:
                scores[category] = recall(node_labels, expected)

        overall = sum(scores.values()) / len(scores)

        print("  Category recall:")
        for cat, score in scores.items():
            bar = "█" * int(score * 20)
            spaces = " " * (20 - len(bar))
            print(f"    {cat:<15} [{bar}{spaces}] {score*100:.0f}%")

        print()
        print(f"  Overall recall: {overall*100:.0f}%")
        print(f"  Nodes extracted: {len(nodes)}")
        print()

        if overall >= 0.80:
            print("  ✅ Target met (≥80%)")
        else:
            print("  ⚠️  Below target. Enable LLM extraction for higher recall.")
            print("     Set: use_llm_extraction: true in tokenmizer.yaml")
        print()


if __name__ == "__main__":
    run()
