#!/usr/bin/env python3
"""
Extraction eval harness.

    python -m benchmarks.eval                      # score the committed corpus
    python -m benchmarks.eval --errors             # list every miss and false positive
    python -m benchmarks.eval --corpus DIR         # score your own labelled sessions
    python -m benchmarks.eval --sweep              # tune a threshold against the corpus
    python -m benchmarks.eval --json OUT.json      # machine-readable output

Reports precision, recall and F1 per category, plus label-quality
statistics that correctness alone does not capture.

Why this exists: the extraction heuristics carry hand-picked constants
(`_is_same_decision`'s 0.82 overlap threshold, the validator's confidence
weights, capture widths). Hand-picked is fine; hand-picked and
*unmeasured* is not, because nobody can tell whether changing one helps.
`--sweep` moves a constant across a range and prints what each value does
to F1, so a number can be defended with a table instead of a feeling.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.eval import corpus as corpus_mod  # noqa: E402
from benchmarks.eval.metrics import (  # noqa: E402
    label_quality,
    score,
)
from tokenmizer.graph_memory.graph import (  # noqa: E402
    GraphMemory,
    NodeStatus,
    NodeType,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ground-truth category -> how to select the matching nodes from the graph.
_SELECTORS = {
    "completed_tasks": lambda n: n.type == NodeType.TASK and n.status == NodeStatus.COMPLETED,
    "pending_tasks":   lambda n: n.type == NodeType.TASK and n.status in (
        NodeStatus.IN_PROGRESS, NodeStatus.PENDING),
    "decisions":       lambda n: n.type == NodeType.DECISION,
    "files":           lambda n: n.type == NodeType.FILE,
    "errors":          lambda n: n.type == NodeType.ERROR,
}


def extract(session) -> dict[str, list[str]]:
    """Run the real extraction pipeline over a session's messages."""
    with tempfile.TemporaryDirectory() as d:
        g = GraphMemory(session.id, storage_dir=d)
        g.extract_from_messages(session.messages, incremental=False)
        nodes = [n for n in g._nodes.values() if not n._evicted]
    return {
        cat: [n.label for n in nodes if sel(n)]
        for cat, sel in _SELECTORS.items()
    }


def evaluate(sessions, threshold: float = 0.6) -> dict:
    per_session, totals = {}, {}
    all_labels: list[str] = []

    for s in sessions:
        got = extract(s)
        scores = {}
        for cat in _SELECTORS:
            want = s.expected(cat)
            if not want and not got[cat]:
                continue          # category not labelled for this session
            sc = score(cat, got[cat], want, threshold)
            scores[cat] = sc
            t = totals.setdefault(cat, {"expected": 0, "extracted": 0,
                                        "tp": 0, "spurious": 0})
            t["expected"] += sc.expected
            t["extracted"] += sc.extracted
            t["tp"] += sc.true_positives
            t["spurious"] += len(sc.spurious)
        per_session[s.id] = scores
        all_labels.extend(x for v in got.values() for x in v)

    micro = {}
    for cat, t in totals.items():
        recall = t["tp"] / t["expected"] if t["expected"] else 1.0
        matched = t["extracted"] - t["spurious"]
        precision = matched / t["extracted"] if t["extracted"] else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        micro[cat] = {"precision": precision, "recall": recall, "f1": f1, **t}

    return {
        "per_session": per_session,
        "micro": micro,
        "labels": label_quality(all_labels),
        "threshold": threshold,
    }


def _bar(v: float, width: int = 18) -> str:
    filled = int(round(v * width))
    return "█" * filled + "·" * (width - filled)


def report(sessions, result: dict, show_errors: bool) -> None:
    print("=" * 74)
    print("TokenMizer — extraction eval")
    print("=" * 74)
    print(corpus_mod.describe(sessions))
    print(f"match threshold: {result['threshold']} token coverage of the expected item")
    print()

    print(f"{'category':<18}{'P':>7}{'R':>7}{'F1':>7}   {'recall':<20}{'n':>5}")
    print("-" * 74)
    macro = []
    for cat, m in result["micro"].items():
        print(f"{cat:<18}{m['precision']:>7.0%}{m['recall']:>7.0%}{m['f1']:>7.0%}   "
              f"{_bar(m['recall']):<20}{m['expected']:>5}")
        macro.append(m["f1"])
    print("-" * 74)
    if macro:
        print(f"{'macro F1':<18}{'':>7}{'':>7}{sum(macro)/len(macro):>7.0%}")
    print()

    # Synthetic vs real, scored separately. A corpus that mixes
    # hand-written fixtures with captured transcripts and reports one
    # number invites exactly the question it cannot answer: does this
    # generalise, or did the heuristics get fitted to the fixtures?
    by_origin = {}
    for s in sessions:
        by_origin.setdefault(s.origin, []).append(s)
    if len(by_origin) > 1:
        print("Generalisation — same extractor, scored by corpus origin")
        print(f"  {'origin':<12}{'sessions':>9}{'macro F1':>10}")
        for origin, group in sorted(by_origin.items()):
            sub = evaluate(group, result["threshold"])
            f1s = [m["f1"] for m in sub["micro"].values()]
            macro = sum(f1s) / len(f1s) if f1s else 0.0
            print(f"  {origin:<12}{len(group):>9}{macro:>10.0%}")
        print()

    q = result["labels"]
    print("Label quality (independent of correctness)")
    print(f"  labels emitted     : {q.count}")
    print(f"  mean length        : {q.mean_chars} chars")
    print(f"  truncated mid-word : {q.truncated} ({q.truncated_pct:.0f}%)")
    print(f"  span >1 sentence   : {q.multi_sentence} ({q.multi_sentence_pct:.0f}%)")
    print(f"  near-duplicate pairs: {q.near_duplicates}")

    if show_errors:
        print()
        print("=" * 74)
        print("Errors")
        print("=" * 74)
        for sid, scores in result["per_session"].items():
            rows = [(c, s) for c, s in scores.items() if s.missed or s.spurious]
            if not rows:
                continue
            print(f"\n{sid}")
            for cat, sc in rows:
                if sc.missed:
                    print(f"  {cat} — MISSED ({len(sc.missed)}):")
                    for m in sc.missed:
                        print(f"      · {m}")
                if sc.spurious:
                    print(f"  {cat} — SPURIOUS ({len(sc.spurious)}):")
                    for m in sc.spurious[:6]:
                        print(f"      · {m[:88]}")
                    if len(sc.spurious) > 6:
                        print(f"      … and {len(sc.spurious) - 6} more")


def sweep(sessions, low: float, high: float, step: float) -> None:
    """Move the match threshold across a range and print the effect.

    This tunes the HARNESS's own matching strictness, which is the first
    thing to pin down: a recall figure means nothing until you know how
    generously 'found' was defined. Extraction constants are swept the
    same way by monkeypatching them around a call to evaluate().
    """
    print("Match-threshold sweep — how strict is 'found'?")
    print(f"{'threshold':>10}{'task R':>9}{'task P':>9}{'dec R':>9}{'dec P':>9}{'macro F1':>10}")
    print("-" * 56)
    t = low
    while t <= high + 1e-9:
        r = evaluate(sessions, threshold=round(t, 2))
        m = r["micro"]
        task = m.get("completed_tasks", {})
        dec = m.get("decisions", {})
        f1s = [v["f1"] for v in m.values()]
        print(f"{t:>10.2f}{task.get('recall', 0):>9.0%}{task.get('precision', 0):>9.0%}"
              f"{dec.get('recall', 0):>9.0%}{dec.get('precision', 0):>9.0%}"
              f"{sum(f1s)/len(f1s) if f1s else 0:>10.0%}")
        t += step


def main() -> int:
    ap = argparse.ArgumentParser(prog="benchmarks.eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", help="directory of labelled session JSON files")
    ap.add_argument("--errors", action="store_true", help="list every miss and false positive")
    ap.add_argument("--threshold", type=float, default=0.6, help="match strictness (default 0.6)")
    ap.add_argument("--sweep", action="store_true", help="sweep the match threshold")
    ap.add_argument("--json", dest="json_out", help="write machine-readable results here")
    args = ap.parse_args()

    logging.disable(logging.CRITICAL)  # extraction logs would drown the report

    try:
        sessions = corpus_mod.load(args.corpus)
    except corpus_mod.CorpusError as e:
        print(f"corpus error: {e}", file=sys.stderr)
        return 2

    if args.sweep:
        sweep(sessions, 0.3, 0.9, 0.1)
        return 0

    result = evaluate(sessions, threshold=args.threshold)
    report(sessions, result, args.errors)

    if args.json_out:
        payload = {
            "corpus": corpus_mod.describe(sessions),
            "threshold": result["threshold"],
            "micro": result["micro"],
            "labels": vars(result["labels"]),
            "per_session": {
                sid: {c: {"precision": s.precision, "recall": s.recall, "f1": s.f1,
                          "missed": s.missed, "spurious": s.spurious}
                      for c, s in scores.items()}
                for sid, scores in result["per_session"].items()
            },
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nSaved: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
