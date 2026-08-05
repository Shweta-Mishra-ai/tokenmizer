#!/usr/bin/env python3
"""
Persistence benchmark — write amplification and concurrent-writer safety.

Measures the two properties the v2 (per-row) storage schema exists to
provide, both of which used to be claimed without a number behind them:

  1. Write amplification. The v1 layout stored one JSON blob per session,
     so every persist rewrote the whole graph — once per chat turn, even
     for a single new node. This measures how many rows a v2 persist
     actually touches, against the v1 baseline of "all of them".

  2. Concurrent-writer safety. Two writers on one session used to mean
     the later save discarded everything the earlier one added. This
     spawns real OS processes (not threads — threads share an
     interpreter and would not exercise the cross-process file lock) and
     counts what survives.

Run:
    python -m benchmarks.persistence.runner

Results are written to benchmark_persistence_results.json.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sqlite3
import statistics
import tempfile
import time
from pathlib import Path

from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType

GRAPH_SIZES = (50, 100, 200)
WRITER_PROCS = 4
NODES_PER_WRITER = 25


def _row_timestamps(db: Path, session: str) -> list[float]:
    conn = sqlite3.connect(str(db))
    try:
        return [r[0] for r in conn.execute(
            "SELECT updated_at FROM graph_nodes WHERE session_id=?", (session,))]
    finally:
        conn.close()


def _seed(storage: str, session: str, n: int) -> GraphMemory:
    g = GraphMemory(session, storage_dir=storage)
    for i in range(n):
        g.add_node(NodeType.TASK,
                   f"Implement feature number {i} in the API layer",
                   NodeStatus.COMPLETED)
    g._persist(force=True)
    return g


def measure_write_amplification() -> list[dict]:
    print("\nWrite amplification — rows touched by one persist")
    print("  graph size   v1 (blob)   v2 (per-row)   reduction")
    rows = []
    for size in GRAPH_SIZES:
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "graph_memory.db"
            g = _seed(d, "amp", size)

            before = max(_row_timestamps(db, "amp"))
            time.sleep(0.01)
            g.add_node(NodeType.DECISION,
                       "Use PostgreSQL for the user database",
                       NodeStatus.COMPLETED)
            g._persist()
            written = sum(1 for t in _row_timestamps(db, "amp") if t > before)

            total = len(g._nodes)
            reduction = 100.0 * (1 - written / total)
            print(f"  {size:>10}   {total:>9}   {written:>12}   {reduction:>8.1f}%")
            rows.append({"graph_nodes": total, "v1_rows_written": total,
                         "v2_rows_written": written,
                         "reduction_pct": round(reduction, 1)})
    return rows


def measure_noop_persist() -> dict:
    """A turn that changes nothing must write nothing."""
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "graph_memory.db"
        _seed(d, "noop", 100)
        time.sleep(0.01)
        mark = max(_row_timestamps(db, "noop"))
        GraphMemory("noop", storage_dir=d)._persist()
        written = sum(1 for t in _row_timestamps(db, "noop") if t > mark)
    print(f"\nNo-op persist: {written} rows written (v1 would rewrite 100)")
    return {"v1_rows_written": 100, "v2_rows_written": written}


def measure_persist_latency() -> dict:
    print("\nPersist latency — one added node on a 200-node graph")
    with tempfile.TemporaryDirectory() as d:
        g = _seed(d, "lat", 200)
        samples = []
        for i in range(30):
            g.add_node(NodeType.TASK, f"Latency probe task number {i} here",
                       NodeStatus.COMPLETED)
            t0 = time.perf_counter()
            g._persist()
            samples.append((time.perf_counter() - t0) * 1000)
    out = {
        "samples": len(samples),
        "median_ms": round(statistics.median(samples), 3),
        "p95_ms": round(sorted(samples)[int(len(samples) * 0.95) - 1], 3),
        "max_ms": round(max(samples), 3),
    }
    print(f"  median {out['median_ms']}ms | p95 {out['p95_ms']}ms | max {out['max_ms']}ms")
    return out


def _writer(storage_dir: str, worker_id: int, count: int, barrier) -> None:
    import logging
    logging.disable(logging.CRITICAL)
    from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType
    g = GraphMemory("shared", storage_dir=storage_dir)
    barrier.wait()
    for i in range(count):
        g.add_node(NodeType.TASK,
                   f"Worker {worker_id} implemented feature {i} in the API",
                   NodeStatus.COMPLETED)
        g._persist()


def measure_concurrent_writers() -> dict:
    print(f"\nConcurrent writers — {WRITER_PROCS} OS processes, one session")
    with tempfile.TemporaryDirectory() as d:
        GraphMemory("shared", storage_dir=d)._persist(force=True)
        barrier = mp.Barrier(WRITER_PROCS)
        procs = [mp.Process(target=_writer, args=(d, w, NODES_PER_WRITER, barrier))
                 for w in range(WRITER_PROCS)]
        t0 = time.perf_counter()
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)
        elapsed = time.perf_counter() - t0

        labels = {n.label for n in GraphMemory("shared", storage_dir=d)._nodes.values()}
        per_worker = {w: sum(1 for x in labels if x.startswith(f"Worker {w} "))
                      for w in range(WRITER_PROCS)}

    expected = WRITER_PROCS * NODES_PER_WRITER
    survived = sum(per_worker.values())
    print(f"  expected {expected} nodes, persisted {survived}")
    for w, got in per_worker.items():
        flag = "ok" if got == NODES_PER_WRITER else f"LOST {NODES_PER_WRITER - got}"
        print(f"    worker {w}: {got}/{NODES_PER_WRITER}  {flag}")
    print(f"  elapsed: {elapsed:.2f}s")
    return {"writers": WRITER_PROCS, "nodes_per_writer": NODES_PER_WRITER,
            "expected": expected, "survived": survived,
            "per_worker": per_worker, "elapsed_s": round(elapsed, 2),
            "lossless": survived == expected}


def measure_stale_writer() -> dict:
    """A writer holding a pre-prune graph must not reinstate the rows
    another writer deleted."""
    print("\nStale writer — one worker prunes, another holds the old view")
    with tempfile.TemporaryDirectory() as d:
        _seed(d, "stale", 100)
        a = GraphMemory("stale", storage_dir=d)
        b = GraphMemory("stale", storage_dir=d)
        a.prune(max_nodes=40)
        a._persist()
        b.add_node(NodeType.DECISION, "Use PostgreSQL for the user database",
                   NodeStatus.COMPLETED)
        b._persist()
        final = GraphMemory("stale", storage_dir=d)
        on_disk = len(final._nodes)
        kept_new = any("PostgreSQL" in n.label for n in final._nodes.values())
    ok = on_disk <= 41 and kept_new
    print(f"  rows on disk: {on_disk} (<=41 expected) | new node kept: {kept_new}")
    print(f"  {'ok' if ok else 'FAILED — stale writer undid the prune'}")
    return {"rows_on_disk": on_disk, "prune_held": on_disk <= 41,
            "new_node_kept": kept_new, "ok": ok}


def run() -> dict:
    print("=" * 62)
    print("TokenMizer — Persistence Benchmark (schema v2, per-row)")
    print("=" * 62)

    results = {
        "write_amplification": measure_write_amplification(),
        "noop_persist": measure_noop_persist(),
        "persist_latency": measure_persist_latency(),
        "concurrent_writers": measure_concurrent_writers(),
        "stale_writer": measure_stale_writer(),
    }

    out = Path("benchmark_persistence_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")
    return results


if __name__ == "__main__":
    run()
