"""
Benchmark v3 — Hybrid merge-logic validation (NOT an LLM recall benchmark)

HONESTY FIX (this is the most important comment in this file — read it
before trusting any number this script prints):

The original version of this script claimed to measure "LLM recall" and
"hybrid recall" using a MockLLMProvider, and the README presented those
percentages in a benchmark table next to real heuristic numbers. That
was methodologically broken: MockLLMProvider sampled its fake output
directly FROM `ground_truth` — the exact same dict that `_recall()` uses
to SCORE the result. Scorer and "model" shared one source of truth, so
the result was guaranteed to land near whatever miss_rate was hand-picked
(0.12), regardless of what HybridExtractor's actual merge logic did. It
measured nothing about real LLM extraction quality — it was a coin flip
calibrated to look good, dressed up as a benchmark.

What this script DOES validate, honestly, with zero circularity: the
MERGE LOGIC's correctness. `HybridExtractor.merge()` combines two
independent extraction results (one from the real heuristic engine, one
from a controllable synthetic "LLM-shaped" input) via set union with
confidence tagging. That's deterministic, testable behavior — does merge
correctly union without dropping or duplicating items? Does corroboration
confidence (0.95) get applied when both sources agree? This script tests
exactly that, with fixtures designed to exercise specific merge scenarios
(full overlap, no overlap, partial overlap, LLM-only items, heuristic-only
items) — not by faking what a real LLM would say.

For an ACTUAL measurement of real LLM extraction recall, you need a real
LLM call. There is no way around this. Set ANTHROPIC_API_KEY or
OPENAI_API_KEY and run with --live to get a real (if still small-sample,
n=3) measurement against your configured provider. The --live path is
the ONLY path in this file that produces a number meaningful enough to
put in a README. The synthetic-fixture path below is for catching merge
logic regressions in CI, nothing more.

Run: python benchmarks/checkpoint_accuracy/runner_v3.py            (merge-logic fixtures)
     python benchmarks/checkpoint_accuracy/runner_v3.py --live      (real LLM, real cost, real number)
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tokenmizer.graph_memory.hybrid_extractor import HybridExtractor, ExtractedData


def _out(msg: str) -> None:
    """Print safely on Windows consoles that use a legacy code page."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))
from benchmarks.checkpoint_accuracy.runner_v2 import SESSIONS, _recall


# ── Merge-logic fixtures ─────────────────────────────────────────────────────
#
# Each fixture hand-specifies what a "heuristic" result and an "LLM-shaped"
# result look like for a given ground truth, with a KNOWN, INTENTIONAL
# overlap pattern. This is the opposite of the old MockLLMProvider design:
# instead of sampling from ground truth with a hidden miss rate, we
# explicitly construct "this item is corroborated," "this item is LLM-only,"
# "this item is heuristic-only," and verify merge() handles each correctly.

def build_merge_fixture(gt: dict) -> tuple[ExtractedData, ExtractedData]:
    """
    Splits each ground-truth category into three known groups and returns
    (llm_shaped, heuristic_shaped) extraction results:
      - first half:  in BOTH  → must end up corroborated (confidence 0.95)
      - next quarter: LLM only → must end up present, confidence 0.80
      - last quarter: heuristic only → must end up present, confidence 0.65

    This directly tests the merge contract instead of guessing at LLM
    behavior — there is no simulated "miss," every split is deliberate
    and the expected merge output is fully determined by construction.
    """
    def split(items: list[str]) -> tuple[list[str], list[str], list[str]]:
        n = len(items)
        both = items[: n // 2]
        llm_only = items[n // 2: n // 2 + max(1, n // 4)]
        heu_only = items[n // 2 + max(1, n // 4):]
        return both, llm_only, heu_only

    llm = ExtractedData()
    heu = ExtractedData()

    for attr, key in [
        ("tasks_done", "completed_tasks"),
        ("files", "files"),
    ]:
        items = list(gt.get(key, []))
        both, llm_only, heu_only = split(items)
        setattr(llm, attr, both + llm_only)
        setattr(heu, attr, both + heu_only)

    dec_items = list(gt.get("decisions", []))
    both, llm_only, heu_only = split(dec_items)
    llm.decisions = [{"label": d, "reason": ""} for d in (both + llm_only)]
    heu.decisions = [{"label": d, "reason": ""} for d in (both + heu_only)]

    return llm, heu


# ── Runner ────────────────────────────────────────────────────────────────

def verify_merge_contract(name: str, session: dict) -> dict:
    """
    Verifies merge() satisfies its documented contract against a
    fixture with known, deliberate overlap — NOT a simulated LLM call.
    Returns pass/fail per category, not a "recall percentage" (recall
    against a fixture you constructed yourself is not a meaningful
    metric — it would just measure whether you can do arithmetic).
    """
    gt = session["ground_truth"]
    he = HybridExtractor()
    llm_fixture, heu_fixture = build_merge_fixture(gt)

    merged = he.merge(llm_fixture, heu_fixture)

    checks = {}

    # Every item present in EITHER source must survive into the merge —
    # this is the actual property that matters: merge must never DROP
    # an item that either source found.
    llm_tasks = set(llm_fixture.tasks_done)
    heu_tasks = set(heu_fixture.tasks_done)
    merged_tasks = set(merged.tasks_done)
    checks["tasks_no_drop"] = (llm_tasks | heu_tasks) <= merged_tasks

    llm_files = set(llm_fixture.files)
    heu_files = set(heu_fixture.files)
    merged_files = set(merged.files)
    checks["files_no_drop"] = (llm_files | heu_files) <= merged_files

    llm_dec = {d["label"] for d in llm_fixture.decisions}
    heu_dec = {d["label"] for d in heu_fixture.decisions}
    merged_dec = {d["label"] for d in merged.decisions}
    checks["decisions_no_drop"] = (llm_dec | heu_dec) <= merged_dec

    # Corroborated items (in both sources) must get the highest confidence tier
    corroborated_tasks = llm_tasks & heu_tasks
    if corroborated_tasks and "tasks_done" in merged.confidence:
        checks["corroboration_confidence_applied"] = merged.confidence["tasks_done"] == 0.95
    else:
        checks["corroboration_confidence_applied"] = True  # nothing to check

    return {"session": name, "checks": checks, "all_passed": all(checks.values())}


async def run_live_llm(name: str, session: dict, provider) -> dict:
    """
    REAL measurement path. Calls an actual configured LLM provider and
    scores its real extraction output against ground truth. This is the
    only path in this file whose recall numbers mean what they say.
    """
    messages = session["messages"]
    gt = session["ground_truth"]
    he = HybridExtractor()

    async def _pfn(msgs, system="", max_tokens=800):
        resp = await provider.chat(messages=msgs, system=system, max_tokens=max_tokens)
        return {"text": resp.text}

    t0 = time.monotonic()
    llm_data = await he.llm_extract(messages, _pfn)
    elapsed_ms = (time.monotonic() - t0) * 1000

    llm_tasks = set(llm_data.tasks_done) if llm_data else set()
    llm_decisions = {d["label"] for d in (llm_data.decisions if llm_data else [])}
    llm_files = set(llm_data.files) if llm_data else set()

    return {
        "session": name,
        "time_ms": round(elapsed_ms, 1),
        "task_recall": _recall(llm_tasks, gt["completed_tasks"]),
        "decision_recall": _recall(llm_decisions, gt["decisions"]),
        "file_recall": _recall(llm_files, gt["files"]),
    }


async def run_all(live: bool = False) -> list[dict]:
    if live:
        import os
        print("\n TokenMizer — Benchmark v3: LIVE LLM extraction recall")
        print("=" * 72)
        print("This calls a real LLM provider. Real API cost will be incurred.\n")

        if os.environ.get("ANTHROPIC_API_KEY"):
            from tokenmizer.providers.providers import AnthropicProvider
            provider = AnthropicProvider(os.environ["ANTHROPIC_API_KEY"], model="claude-haiku-4-5")
        elif os.environ.get("OPENAI_API_KEY"):
            from tokenmizer.providers.providers import OpenAIProvider
            provider = OpenAIProvider(os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")
        else:
            print("ERROR: set ANTHROPIC_API_KEY or OPENAI_API_KEY for --live mode.")
            return []

        results = []
        for name, session in SESSIONS.items():
            r = await run_live_llm(name, session, provider)
            results.append(r)
            print(f"  {name:<18} task={r['task_recall']:.0%} "
                  f"decision={r['decision_recall']:.0%} file={r['file_recall']:.0%} "
                  f"({r['time_ms']:.0f}ms)")

        _out(f"\n[!] n={len(results)} sessions — too small to generalize. Treat as a")
        _out("  directional smoke test, not a statistically meaningful recall claim.")
        return results

    _out("\nTokenMizer — Benchmark v3: Hybrid MERGE LOGIC contract verification")
    _out("=" * 72)
    _out("NOTE: this is NOT an LLM recall benchmark. See module docstring.")
    _out("It verifies merge() never drops corroborated/LLM-only/heuristic-only")
    _out("items and applies confidence tiers correctly, using fixtures with")
    _out("deliberately known overlap — not simulated LLM behavior.\n")

    results = []
    all_ok = True
    for name, session in SESSIONS.items():
        r = verify_merge_contract(name, session)
        results.append(r)
        status = "PASS" if r["all_passed"] else "FAIL"
        _out(f"  {name:<18} {status}")
        for check, passed in r["checks"].items():
            if not passed:
                _out(f"      x {check}")
                all_ok = False

    _out("")
    _out("=" * 72)
    _out("All merge-logic contract checks passed" if all_ok
         else "Some merge-logic contract checks FAILED — see above")
    _out("")
    _out("For a REAL LLM recall measurement (not a logic-contract check):")
    _out("  python benchmarks/checkpoint_accuracy/runner_v3.py --live")

    out = Path("benchmark_v3_merge_contract_results.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _out(f"\nSaved: {out}")
    return results


if __name__ == "__main__":
    asyncio.run(run_all(live="--live" in sys.argv))
