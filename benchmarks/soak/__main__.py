"""
Long-session soak: does the proxy stay fast, bounded and lossless when a
session gets long and large?

    python -m benchmarks.soak                       # 1,000 turns, defaults below
    python -m benchmarks.soak --turns 3000 --giant-mb 8
    python -m benchmarks.soak --concurrent 8 --turns 300
    python -m benchmarks.soak --json out.json

What runs: the real proxy (FastAPI app, windowing, compression, graph
extraction, SQLite persistence, recall), driven the way a chat client drives
it — the whole history resent on every request. Only the network is
replaced: the Anthropic SDK client is a stand-in that records exactly what
the proxy would have sent and answers "ok". Nothing in TokenMizer is
stubbed.

The session: filler text from the captured real transcripts, tool outputs
of 1-200 KB, one giant tool output (--giant-mb), and 50 planted facts
spread over the first 60% of the session — 10 each of decisions, errors,
pending tasks, edited files and finished tasks. Each fact carries one answer
token that appears nowhere else, so "is it still there" has an exact answer.

Measured:

  latency      wall time per request, first 10% vs last 10% of the session,
               and the largest single request
  memory       process RSS growth over the session, and the graph's size
  graph        each planted fact's answer token stored in the graph
  recall       a question about the fact, asked at the END of the session
               (long after its turn left the window): is the answer token in
               what the proxy sends upstream for that question?
  restart      the graph reloaded from disk by a fresh GraphMemory: are the
               facts still there?
  tail         the client's newest message reaches upstream byte-identical
  isolation    (--concurrent N) N sessions in parallel; a question in one
               session must never receive another session's facts

HONEST LIMITS:
  * The stand-in answers "ok", so model behaviour is not measured — only
    what the model would have been GIVEN.
  * Facts are planted in explicit wording ("Decision: ...", "TODO: ...").
    Extraction of implicit phrasing is what benchmarks.eval and the
    research held-out corpora measure; this measures what happens to facts
    once extracted, over length.
  * One process, in-memory test transport: no real network, no second
    worker.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_ADJ = ["amber", "basalt", "cobalt", "delta", "ember", "fjord", "garnet", "harbor",
        "indigo", "juniper", "kestrel", "lumen", "marble", "nimbus", "onyx", "pylon",
        "quartz", "raven", "sable", "tundra"]
_NOUN = ["ledger", "gateway", "scheduler", "indexer", "billing", "exporter", "importer",
         "notifier", "uploader", "resolver"]


def _rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _facts(rng: random.Random, tag: str) -> list[dict]:
    """50 facts: (kind, subject, answer token, the turn text, a question)."""
    subjects = [f"{a} {n}" for a in _ADJ for n in _NOUN]
    rng.shuffle(subjects)
    out = []
    for i in range(50):
        kind = ["decision", "error", "pending", "file", "done"][i % 5]
        subj = subjects[i]
        slug = subj.replace(" ", "_")
        ans = f"{slug}_{tag}{i:02d}"
        if kind == "decision":
            role, text = "assistant", (
                f"Decision: for the {subj} we'll use `{ans}` as the partition key, "
                f"because it keeps ordering per tenant and Postgres row locks were too slow.")
            q = f"What did we decide for the {subj}?"
        elif kind == "error":
            role, text = "tool", (
                f"Traceback (most recent call last):\n"
                f"  File \"src/{slug}/handler.py\", line 42, in run\n"
                f"    self.flush()\nRuntimeError: {ans} overflowed the batch buffer")
            q = f"What error did the {subj} handler hit?"
        elif kind == "pending":
            role, text = "user", (
                f"TODO: we still need to add the `{ans}` retry budget to the {subj} "
                f"before Friday's release.")
            q = f"What is still pending for the {subj}?"
        elif kind == "file":
            role, text = "assistant", (
                f"I edited `src/{slug}/{ans}.py` to add the missing guard for the {subj}.")
            q = f"Which file did we change for the {subj}?"
        else:
            role, text = "assistant", (
                f"Done: migrated the {subj} to the `{ans}` schema and all its tests pass.")
            q = f"What did we finish on the {subj}?"
        out.append({"kind": kind, "subject": subj, "answer": ans, "role": role,
                    "text": text, "question": q})
    return out


def _filler() -> list[tuple[str, str]]:
    from benchmarks.eval.corpus import load
    pairs = []
    for s in load():
        ms = [m for m in s.messages if isinstance(m.get("content"), str)]
        pairs += [(ms[i]["content"], ms[i + 1]["content"]) for i in range(0, len(ms) - 1, 2)
                  if ms[i]["role"] == "user" and ms[i + 1]["role"] == "assistant"]
    return pairs


def _tool_output(rng: random.Random, kb: int, turn: int) -> str:
    """A directory listing / log dump of about `kb` KB: realistic, repetitive."""
    lines, size = [], 0
    while size < kb * 1024:
        n = rng.randint(1, 9999)
        line = (f"{turn:05d}:{n:04d}  src/pkg_{n % 37}/mod_{n % 101}.py   "
                f"{rng.randint(100, 99999):>6} bytes  ok  checksum={rng.getrandbits(64):016x}")
        lines.append(line)
        size += len(line) + 1
    return "\n".join(lines)


def _build_session(turns: int, giant_mb: float, rng: random.Random, tag: str):
    """A list of client turns. Each entry is a list of messages appended to
    the history before the next request (the last one is a user or tool turn)."""
    filler = _filler()
    facts = _facts(rng, tag)
    plant_at = {int(turns * 0.6 * (i + 1) / (len(facts) + 1)): f for i, f in enumerate(facts)}
    giant_at = int(turns * 0.75) if giant_mb > 0 else -1
    steps: list[list[dict]] = []
    call = 0
    for t in range(turns):
        user, assistant = filler[t % len(filler)]
        step: list[dict] = []
        f = plant_at.get(t)
        if f and f["role"] == "assistant":
            step.append({"role": "assistant", "content": f["text"]})
        if t % 4 == 3 or t == giant_at or (f and f["role"] == "tool"):
            call += 1
            cid = f"call_{tag}_{call}"
            if t == giant_at:
                out = _tool_output(rng, int(giant_mb * 1024), t)
            elif f and f["role"] == "tool":
                out = _tool_output(rng, 2, t) + "\n" + f["text"]
            else:
                out = _tool_output(rng, rng.choice([1, 4, 16, 64, 200]), t)
            step.append({"role": "assistant", "content": "",
                         "tool_calls": [{"id": cid, "type": "function",
                                         "function": {"name": "bash",
                                                      "arguments": json.dumps({"cmd": "ls -R"})}}]})
            step.append({"role": "tool", "tool_call_id": cid, "content": out})
            step.append({"role": "assistant", "content": assistant})
        if f and f["role"] == "user":
            step.append({"role": "user", "content": f["text"]})
        else:
            step.append({"role": "user", "content": f"[turn {t}] {user}"})
        steps.append(step)
    return steps, facts, giant_at


class _Upstream:
    """Stand-in for anthropic.AsyncAnthropic: keeps the LAST request it was
    sent and a count. Keeping every request would make the process's memory
    growth measure this harness instead of the proxy."""

    def __init__(self):
        self.count = 0
        self.last: dict = {"text": "", "messages": []}
        self.lock = threading.Lock()

    def client_class(self):
        upstream = self

        class Messages:
            async def create(self, *, model, messages, max_tokens, system=None, **kw):
                parts = []
                for b in ([{"text": system}] if isinstance(system, str) else system or []):
                    parts.append(b.get("text", ""))
                for m in messages:
                    c = m["content"]
                    for b in ([{"text": c}] if isinstance(c, str) else c):
                        parts.append(b.get("text") or json.dumps(
                            b.get("content", b.get("input", "")), default=str))
                text = "\n".join(parts)
                with upstream.lock:
                    upstream.count += 1
                    upstream.last = {"text": text, "messages": messages}
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="ok")], stop_reason="end_turn",
                    usage=SimpleNamespace(input_tokens=len(text) // 4, output_tokens=1,
                                          cache_read_input_tokens=0,
                                          cache_creation_input_tokens=0))

        class Client:
            def __init__(self, *a, **k):
                self.messages = Messages()

        return Client


def _setup(model: str):
    storage = tempfile.mkdtemp(prefix="tokenmizer-soak-")
    os.environ["TOKENMIZER_GRAPH_CHECKPOINT__STORAGE_DIR"] = storage
    import anthropic

    from tokenmizer.api import app as A
    from tokenmizer.providers.providers import AnthropicProvider

    up = _Upstream()
    anthropic.AsyncAnthropic = up.client_class()
    provider = AnthropicProvider(api_key="soak", model=model)
    A._get_provider = lambda: provider

    async def _unlimited(request):
        return None

    A._check_rate_limit = _unlimited
    A.settings.provider = "anthropic"
    A.settings.cache.enabled = False
    A.settings.api_key = ""
    A.settings.graph_checkpoint.storage_dir = storage
    A._graph_cache.clear()
    A._smart_window._frozen.clear()
    return A, up, storage


def _graph_text(g) -> str:
    return "\n".join(f"{n.label} {getattr(n, 'content', '') or ''} "
                     f"{json.dumps(getattr(n, 'metadata', {}) or {}, default=str)}"
                     for n in g._nodes.values() if not n._evicted)


def _run_session(c, A, up, sid, steps, model, latencies, rss, tail_ok, lock=None):
    history: list[dict] = []
    for i, step in enumerate(steps):
        history.extend(step)
        before = up.count
        t = time.perf_counter()
        r = c.post("/v1/chat/completions",
                   json={"model": model, "messages": history, "session_id": sid})
        dt = time.perf_counter() - t
        if r.status_code != 200:
            raise SystemExit(f"{sid} turn {i}: proxy returned {r.status_code}: {r.text[:300]}")
        latencies.append(dt)
        if lock is None:
            # The newest client message must reach upstream untouched.
            newest = history[-1]["content"]
            if up.count > before and newest not in up.last["text"]:
                tail_ok.append(i)
            if i % 50 == 0:
                rss.append((i, _rss_mb()))
            if i % 100 == 0:
                print(f"  {sid}: request {i}/{len(steps)}, {dt * 1000:.0f} ms, "
                      f"RSS {_rss_mb():.0f} MB", file=sys.stderr, flush=True)
    return history


def _probe(c, up, sid, history, model, question) -> str:
    probe = history + [{"role": "user", "content": question}]
    with up.lock:
        before = up.count
    r = c.post("/v1/chat/completions",
               json={"model": model, "messages": probe, "session_id": sid})
    if r.status_code != 200 or up.count == before:
        return ""
    return up.last["text"]


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def single(args) -> dict:
    from fastapi.testclient import TestClient

    from tokenmizer.graph_memory.graph import GraphMemory

    A, up, storage = _setup(args.model)
    rng = random.Random(args.seed)
    steps, facts, giant_at = _build_session(args.turns, args.giant_mb, rng, "s")
    sid = "soak"
    latencies: list[float] = []
    rss: list[tuple[int, float]] = []
    tail_bad: list[int] = []
    rss0 = _rss_mb()
    t0 = time.perf_counter()
    with TestClient(A.app) as c:
        history = _run_session(c, A, up, sid, steps, args.model, latencies, rss, tail_bad)
        run_s = time.perf_counter() - t0
        history_mb = len(json.dumps(history)) / 1e6
        last_sent_mb = len(up.last["text"]) / 1e6
        g = A._graph_cache.get(sid)
        gtext = _graph_text(g) if g else ""
        nodes = sum(1 for n in g._nodes.values() if not n._evicted) if g else 0
        recall = {}
        for f in facts:
            recall[f["answer"]] = f["answer"] in _probe(c, up, sid, history, args.model,
                                                        f["question"])
    # Restart: a fresh GraphMemory loads the session from disk.
    A._graph_cache.clear()
    g2 = GraphMemory(sid, storage_dir=storage)
    rtext = _graph_text(g2)
    rss_end = _rss_mb()

    n10 = max(1, len(latencies) // 10)
    by_kind = {}
    for f in facts:
        k = by_kind.setdefault(f["kind"], {"n": 0, "graph": 0, "recall": 0, "restart": 0})
        k["n"] += 1
        k["graph"] += f["answer"] in gtext
        k["recall"] += recall[f["answer"]]
        k["restart"] += f["answer"] in rtext
    db = sum(p.stat().st_size for p in Path(storage).rglob("*") if p.is_file()) / 1e6
    return {
        "turns": args.turns, "requests": len(latencies), "run_seconds": round(run_s, 1),
        "history_mb_at_end": round(history_mb, 2), "upstream_mb_last_request": round(last_sent_mb, 3),
        "giant_turn": giant_at, "giant_mb": args.giant_mb,
        "latency_ms": {
            "first10pct_p50": round(st.median(latencies[:n10]) * 1000, 1),
            "last10pct_p50": round(st.median(latencies[-n10:]) * 1000, 1),
            "last10pct_p95": round(_pct(latencies[-n10:], 0.95) * 1000, 1),
            "max": round(max(latencies) * 1000, 1),
            "max_at_turn": latencies.index(max(latencies)),
        },
        "rss_mb": {"start": round(rss0, 1), "end": round(rss_end, 1),
                   "samples": [(i, round(m, 1)) for i, m in rss[:: max(1, len(rss) // 10)]]},
        "graph_nodes": nodes, "storage_mb": round(db, 2),
        "facts": by_kind,
        "facts_total": {"n": len(facts),
                        "graph": sum(v["graph"] for v in by_kind.values()),
                        "recall": sum(v["recall"] for v in by_kind.values()),
                        "restart": sum(v["restart"] for v in by_kind.values())},
        "lost_in_graph": [f["answer"] for f in facts if f["answer"] not in gtext],
        "not_recalled": [f["answer"] for f in facts if not recall[f["answer"]]],
        "tail_altered_at": tail_bad[:10], "tail_altered_count": len(tail_bad),
    }


def concurrent(args) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    from fastapi.testclient import TestClient

    A, up, _storage = _setup(args.model)
    sessions = {}
    for k in range(args.concurrent):
        rng = random.Random(args.seed + k)
        steps, facts, _ = _build_session(args.turns, 0, rng, f"c{k}x")
        sessions[f"soak-{k}"] = (steps, facts)
    latencies: list[float] = []
    histories = {}
    t0 = time.perf_counter()
    with TestClient(A.app) as c:
        def go(sid):
            lat: list[float] = []
            h = _run_session(c, A, up, sid, sessions[sid][0], args.model, lat, [], [], lock=True)
            return sid, h, lat
        with ThreadPoolExecutor(args.concurrent) as ex:
            for sid, h, lat in ex.map(go, list(sessions)):
                histories[sid] = h
                latencies += lat
        run_s = time.perf_counter() - t0
        own = foreign = leaks = 0
        leak_examples = []
        missed = []
        sids = list(sessions)
        for i, sid in enumerate(sids):
            other = sids[(i + 1) % len(sids)]
            for f in sessions[sid][1][:10]:
                hit = f["answer"] in _probe(c, up, sid, histories[sid], args.model, f["question"])
                own += hit
                if not hit:
                    missed.append((sid, f["kind"], f["answer"]))
            # The same question shape about the OTHER session's subjects.
            for f in sessions[other][1][:10]:
                foreign += 1
                sent = _probe(c, up, sid, histories[sid], args.model, f["question"])
                if f["answer"] in sent:
                    leaks += 1
                    if len(leak_examples) < 5:
                        leak_examples.append((sid, f["answer"]))
    return {"sessions": args.concurrent, "turns_each": args.turns,
            "requests": len(latencies), "run_seconds": round(run_s, 1),
            "latency_ms_p50": round(st.median(latencies) * 1000, 1),
            "latency_ms_p95": round(_pct(latencies, 0.95) * 1000, 1),
            "own_recall": f"{own}/{10 * args.concurrent}", "own_missed": missed,
            "cross_session_leaks": f"{leaks}/{foreign}", "leak_examples": leak_examples}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--turns", type=int, default=1000)
    ap.add_argument("--giant-mb", type=float, default=4.0,
                    help="size of one giant tool output at 75%% of the session (0 = none)")
    ap.add_argument("--concurrent", type=int, default=0,
                    help="run N sessions in parallel instead, and check isolation")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    import logging
    logging.disable(logging.WARNING)
    res = concurrent(args) if args.concurrent else single(args)
    print(json.dumps(res, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
