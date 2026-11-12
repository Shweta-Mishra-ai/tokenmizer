"""
Rank the Ollama models installed on THIS machine as session-memory extractors.

    python -m benchmarks.extraction_models                    # every installed chat model
    python -m benchmarks.extraction_models --models qwen2.5:7b,llama3.1:8b
    python -m benchmarks.extraction_models --base-url http://host:11434 --json out.json

For each model, every session of the labelled extraction corpus
(benchmarks/eval/corpus) goes through exactly the call the proxy makes in
the background (graph_memory/llm_extraction.extraction_fn: schema-constrained
JSON, temperature 0, thinking off), and the result is applied to a graph the
way the proxy applies it. Two scores per model:

  model alone       what the model extracted, on its own
  model+heuristic   merged with the always-on heuristic pass — what the graph
                    actually holds in production

plus the share of calls that did not return usable JSON and seconds per call
on your hardware. The heuristic-only row is the baseline: a model is worth
running only if it beats that.

Sessions are sent in chunks of up to 20 messages, because llm_extract reads
at most the last 20 of whatever it is given.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time

from benchmarks.eval import corpus as corpus_mod
from benchmarks.eval.__main__ import _SELECTORS
from benchmarks.eval.metrics import score
from tokenmizer.graph_memory.graph import GraphMemory
from tokenmizer.graph_memory.hybrid_extractor import ExtractedData, HybridExtractor
from tokenmizer.graph_memory.llm_extraction import extraction_fn
from tokenmizer.providers.ollama_models import installed_models, pick_model
from tokenmizer.providers.providers import OllamaProvider

_CHUNK = 20
_CATEGORIES = list(_SELECTORS)


def _labels_from(session, extracted) -> dict[str, list[str]]:
    """Apply `extracted` to a fresh graph and read the categories back,
    as benchmarks/eval does for the heuristic pass."""
    with tempfile.TemporaryDirectory() as d:
        g = GraphMemory(session.id, storage_dir=d)
        if extracted is not None:
            g.extract_from_messages(session.messages, incremental=False,
                                    extracted_data=extracted)
        nodes = [n for n in g._nodes.values() if not n._evicted]
        return {cat: [n.label for n in nodes if sel(n)] for cat, sel in _SELECTORS.items()}


def _merge_chunks(parts: list) -> ExtractedData | None:
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    out = ExtractedData()
    for p in parts:
        for name, value in vars(p).items():
            if isinstance(value, list):
                getattr(out, name).extend(value)
    return out


def _score(sessions, got_by_session) -> dict:
    totals = {c: {"expected": 0, "extracted": 0, "tp": 0, "spurious": 0} for c in _CATEGORIES}
    for s in sessions:
        got = got_by_session[s.id]
        for cat in _CATEGORIES:
            want = s.expected(cat)
            if not want and not got[cat]:
                continue
            sc = score(cat, got[cat], want)
            t = totals[cat]
            t["expected"] += sc.expected
            t["extracted"] += sc.extracted
            t["tp"] += sc.true_positives
            t["spurious"] += len(sc.spurious)
    f1 = {}
    for cat, t in totals.items():
        recall = t["tp"] / t["expected"] if t["expected"] else 1.0
        precision = (t["extracted"] - t["spurious"]) / t["extracted"] if t["extracted"] else 1.0
        f1[cat] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f1["macro"] = sum(f1[c] for c in _CATEGORIES) / len(_CATEGORIES)
    return f1


async def _run_model(model: str, base_url: str, num_ctx: int, timeout: float, sessions) -> dict:
    provider = OllamaProvider(model, base_url=base_url)
    provider.num_ctx = num_ctx
    provider.request_timeout = timeout
    provider._retry_delays = []          # a benchmark wants the failure, not a retry
    call = extraction_fn(provider)
    extractor = HybridExtractor()
    alone, merged, calls, failures, seconds = {}, {}, 0, 0, 0.0
    for s in sessions:
        parts = []
        for i in range(0, len(s.messages), _CHUNK):
            chunk = s.messages[i:i + _CHUNK]
            t0 = time.perf_counter()
            result = await extractor.llm_extract(chunk, call)
            seconds += time.perf_counter() - t0
            calls += 1
            failures += result is None
            parts.append(result)
        llm = _merge_chunks(parts)
        alone[s.id] = _labels_from(s, llm)
        merged[s.id] = _labels_from(s, extractor.merge(llm, extractor.heuristic_extract(s.messages)))
        print(f"    {s.id}: {calls} calls so far, {failures} without JSON", file=sys.stderr)
    return {"model": model, "alone": _score(sessions, alone), "merged": _score(sessions, merged),
            "calls": calls, "json_failures": failures,
            "seconds_per_call": seconds / calls if calls else 0.0}


def _baseline(sessions) -> dict:
    extractor = HybridExtractor()
    merged = {s.id: _labels_from(s, extractor.heuristic_extract(s.messages)) for s in sessions}
    f1 = _score(sessions, merged)
    return {"model": "(heuristic only)", "alone": f1, "merged": f1,
            "calls": 0, "json_failures": 0, "seconds_per_call": 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", default="", help="comma-separated; default: every installed chat model")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--corpus", default=None, help="labelled corpus directory (default: the committed one)")
    ap.add_argument("--json", default="", help="write the results here as JSON")
    args = ap.parse_args()

    sessions = corpus_mod.load(args.corpus)
    installed = installed_models(args.base_url)
    if installed is None:
        print(f"Ollama did not answer at {args.base_url}. Start it with: ollama serve",
              file=sys.stderr)
        return 2
    models = [m.strip() for m in args.models.split(",") if m.strip()] or [
        m for m in installed if pick_model([m])[0]]
    if not models:
        print(f"No chat model installed. Try: ollama pull {pick_model([])[1].split()[-1]}",
              file=sys.stderr)
        return 2

    rows = [_baseline(sessions)]
    for m in models:
        print(f"  {m} ...", file=sys.stderr)
        rows.append(asyncio.run(_run_model(m, args.base_url, args.num_ctx, args.timeout, sessions)))

    head = f"{'model':28s} {'alone F1':>9s} {'+heuristic':>11s} {'no JSON':>8s} {'s/call':>7s}"
    print(f"\n{len(sessions)} labelled sessions · macro F1 over {', '.join(_CATEGORIES)}\n")
    print(head)
    print("-" * len(head))
    for r in sorted(rows, key=lambda r: r["merged"]["macro"], reverse=True):
        fail = f"{r['json_failures']}/{r['calls']}" if r["calls"] else "-"
        print(f"{r['model']:28s} {r['alone']['macro']:9.1%} {r['merged']['macro']:11.1%} "
              f"{fail:>8s} {r['seconds_per_call']:7.1f}")
    best = max(rows[1:], key=lambda r: (r["merged"]["macro"], -r["seconds_per_call"]), default=None)
    if best and best["merged"]["macro"] > rows[0]["merged"]["macro"]:
        print(f"\nBest here: {best['model']}. Set graph_checkpoint.extraction_model: {best['model']}")
    else:
        print("\nNo installed model beat heuristic-only extraction on this corpus; "
              "leave use_llm_extraction off, or try a larger model.")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
