"""
Scores GraphMemory.query() — the ranker on the request path.

benchmarks/graph_retrieval/runner.py measures what reaches the graph.
Nothing measured what comes back out of it, which matters more: query() is
what api/app.py calls on every turn to decide which nodes get injected into
the prompt, and what SmartMessageWindow leans on once older turns are dropped.

The queries below are deliberately NOT phrased in the node's own words. A
person asking a follow-up says "what are we storing data in", not "PostgreSQL
for order storage" — if they used the node's wording they would not need to
ask. That is precisely the case token-overlap ranking cannot serve, so it is
the case worth measuring.

Ground truth is the substring a correct answer must contain, taken from the
transcripts in benchmarks/eval/corpus. Run:

    python -m benchmarks.graph_retrieval.query_eval
    python -m benchmarks.graph_retrieval.query_eval --semantic
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from tokenmizer.graph_memory.graph import GraphMemory

CORPUS = Path(__file__).resolve().parents[2] / "benchmarks" / "eval" / "corpus"

# (corpus session, question a person would actually type, substring the
#  right node must contain). Paraphrased on purpose — see the module docstring.
CASES: list[tuple[str, str, str]] = [
    ("go_microservice",  "what are we storing orders in",          "postgres"),
    ("go_microservice",  "how do the services talk to each other",  "grpc"),
    ("go_microservice",  "why did the integration test never finish", "timeout"),
    ("go_microservice",  "what did we do about retries",            "retry"),
    ("fastapi_auth",     "how are passwords protected",             "bcrypt"),
    ("fastapi_auth",     "where do refresh tokens live",            "redis"),
    ("fastapi_auth",     "which database are we on",                "postgres"),
    ("frontend_perf",    "what did we do about the bundle size",    "splitting"),
    ("frontend_perf",    "which date library are we on now",        "date-fns"),
    ("data_migration",   "how are we keeping both databases in sync", "replication"),
    ("react_dashboard",  "what is slow about the dashboard",        "render"),
    ("ml_pipeline",      "which model are we fine-tuning",          "distilbert"),
    ("ml_pipeline",      "how do we know if the model is any good", "evaluate"),
]

# Every question above must be answerable from the session's own transcript.
# A question the transcript never answers measures nothing about the ranker —
# it measures extraction, and it will read as a retrieval failure forever. One
# draft asked ml_pipeline "how are we tracking experiments" when that session
# never mentions experiment tracking at all.


def _build(session: str, storage_dir: str) -> GraphMemory | None:
    path = CORPUS / f"{session}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    graph = GraphMemory(session_id=f"queryeval-{session}", storage_dir=storage_dir)
    graph.extract_from_messages(data["messages"], incremental=False)
    return graph


def run(top_k: int = 6) -> float:
    import tempfile

    hits = misses = 0
    graphs: dict[str, GraphMemory | None] = {}
    storage = tempfile.mkdtemp(prefix="tokenmizer-queryeval-")

    print(f"GraphMemory.query() — recall@{top_k} on paraphrased questions\n")
    for session, question, expected in CASES:
        if session not in graphs:
            graphs[session] = _build(session, storage)
        graph = graphs[session]
        if graph is None:
            continue

        results = graph.query(question, top_k=top_k)
        blob = " | ".join(
            f"{n.label} {n.summary or ''}" for n in results
        ).lower()
        ok = expected.lower() in blob
        hits += ok
        misses += not ok
        print(f"  {'hit ' if ok else 'MISS'}  {session:18} {question:44} -> {expected}")
        if not ok:
            top = ", ".join(n.label[:34] for n in results[:3]) or "(nothing returned)"
            print(f"          returned: {top}")

    total = hits + misses
    recall = hits / total if total else 0.0
    print(f"\n  recall@{top_k}: {recall:.0%}  ({hits}/{total})")
    return recall


if __name__ == "__main__":
    if "--semantic" in sys.argv:
        from tokenmizer.config.settings import get_settings
        get_settings().graph_checkpoint.semantic_retrieval = True
        print("(semantic retrieval enabled)\n")
    run()
