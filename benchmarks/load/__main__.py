"""
Load test: the real server, real HTTP, many sessions at once — and every
response checked.

    python -m benchmarks.load                         # 1 worker, 24 sessions x 40 turns
    python -m benchmarks.load --workers 4 --sessions 48 --turns 30
    python -m benchmarks.load --json out.json

What runs: `uvicorn tokenmizer.api.app:app` in its own process (one or more
workers), with provider=ollama pointed at a local stand-in that speaks the
Ollama HTTP API (/api/tags, /api/chat, streaming and not). So the server,
its HTTP stack, the Ollama adapter and every layer of the proxy run as
deployed; only the model is fake. Rate limiting is turned off: this
measures capacity, and the limiter has its own tests.

Each client session sends its whole history on every request, as a chat
client does. Every request is checked, not sampled:

  responses   status 200; OpenAI chat.completion shape; the content is the
              reply the stand-in gave; streams are well-formed SSE ending in
              [DONE] and carry the same reply
  upstream    (checked inside the stand-in, on every request it receives)
              the client's newest message is there verbatim, and no other
              session's marker is — sessions never leak into each other
  hostile     malformed requests get a 4xx, never a 5xx, and never take the
              server down: requests after them still succeed

Measured: throughput, latency percentiles, the server's resident memory
before and after, and every failure with its reason.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import signal
import socket
import statistics as st
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

REPLY = "Understood, proceeding with the next step."
HOSTILE_TAG = "HOSTILE-PROBE"
MARK = re.compile(r"SESSMARK(\d{4})Z")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _Upstream:
    """A stand-in Ollama that validates what the proxy sends it."""

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = 0
        self.violations: list[str] = []
        self.expected: dict[str, str] = {}      # session marker -> newest message
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._json(200, {"models": [{"name": "llama3.1:8b"}]})
                else:
                    self._json(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer._check(body)
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    words = REPLY.split(" ")
                    for i, w in enumerate(words):
                        piece = w if i == len(words) - 1 else w + " "
                        self._chunk(json.dumps({"message": {"role": "assistant", "content": piece},
                                                "done": False}) + "\n")
                    self._chunk(json.dumps({"message": {"role": "assistant", "content": ""},
                                            "done": True, "done_reason": "stop",
                                            "prompt_eval_count": 100, "eval_count": 8}) + "\n")
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    self._json(200, {"message": {"role": "assistant", "content": REPLY},
                                     "done": True, "done_reason": "stop",
                                     "prompt_eval_count": 100, "eval_count": 8})

            def _chunk(self, text):
                data = text.encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", _free_port()), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _check(self, body):
        # The texts themselves, not their JSON encoding: json.dumps escapes
        # newlines, and a multi-line message is then never found.
        parts = []
        for m in body.get("messages", []):
            c = m.get("content")
            parts.append(c if isinstance(c, str) else json.dumps(c, ensure_ascii=False))
        text = "\n".join(parts)
        marks = set(MARK.findall(text))
        with self.lock:
            self.requests += 1
            if body.get("format"):
                return                  # a background extraction call has no client turn
            if HOSTILE_TAG in text:
                return                  # a malformed request the proxy chose to forward
            # Which session is this? The one whose newest message it carries.
            owners = [m for m, newest in self.expected.items() if newest in text]
            if not owners:
                self.violations.append("upstream request carried no client's newest message")
                return
            mine = MARK.search(owners[0]).group(1)
            foreign = marks - {mine}
            if foreign:
                self.violations.append(
                    f"session {mine} request carried other sessions' markers {sorted(foreign)[:3]}")

    def close(self):
        self.server.shutdown()


def _start_server(upstream_url: str, port: int, workers: int, storage: str) -> subprocess.Popen:
    env = dict(os.environ)
    env.update({
        "TOKENMIZER_PROVIDER": "ollama",
        "TOKENMIZER_OLLAMA_BASE_URL": upstream_url,
        "TOKENMIZER_DEFAULT_MODEL": "llama3.1:8b",
        "TOKENMIZER_RATE_LIMIT__ENABLED": "false",
        "TOKENMIZER_GRAPH_CHECKPOINT__STORAGE_DIR": storage,
        "TOKENMIZER_API_KEY": "",
    })
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "tokenmizer.api.app:app", "--host", "127.0.0.1",
         "--port", str(port), "--workers", str(workers), "--log-level", "warning"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code < 500:
                return proc
        except httpx.HTTPError:
            pass
        if proc.poll() is not None:
            raise SystemExit("server exited at startup:\n" + proc.stderr.read().decode()[-2000:])
        time.sleep(0.3)
    raise SystemExit("server did not come up within 60 s")


def _rss_mb(pid: int) -> float:
    """Resident memory of the server process and its children (workers)."""
    total = 0
    pids = [pid]
    try:
        pids += [int(p) for p in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except OSError:
        pass
    for p in pids:
        try:
            for line in Path(f"/proc/{p}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
        except OSError:
            pass
    return total / 1024


def _user_turn(rng: random.Random, sid: int, t: int) -> str:
    kinds = [
        f"Turn {t}: let's continue with the export step for session part {t}.",
        f"Turn {t}: Decision: use `queue_{sid}_{t}` as the partition key for this stage.",
        f"Turn {t}: TODO: write tests for the handler in src/svc_{sid}/mod_{t}.py",
        f"Turn {t}: the job failed with RuntimeError: overflow in batch {t}",
        f"Turn {t}: " + "log line ok\n" * rng.choice([5, 50, 400]),
    ]
    return rng.choice(kinds)


async def _session(client, base, up: _Upstream, sid: int, turns: int, stream_share: float,
                   lat: list, failures: list, rng: random.Random):
    marker = f"SESSMARK{sid:04d}Z"
    history = [{"role": "user", "content": f"We are building the {marker} service. Start."}]
    for t in range(turns):
        if t:
            # Tagged with the session so the stand-in can tell whose request
            # it is; the tag is not the marker the leak check looks for.
            history.append({"role": "user", "content": f"[client {sid}] " + _user_turn(rng, sid, t)})
        newest = history[-1]["content"]
        with up.lock:
            up.expected[marker] = newest
        stream = rng.random() < stream_share
        body = {"model": "llama3.1:8b", "messages": history, "session_id": f"load-{sid}",
                "stream": stream, "max_tokens": 200}
        t0 = time.perf_counter()
        try:
            if stream:
                text, ok = "", False
                async with client.stream("POST", f"{base}/v1/chat/completions", json=body,
                                         timeout=120) as r:
                    if r.status_code != 200:
                        failures.append(f"s{sid} t{t}: stream status {r.status_code}")
                        return
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            ok = True
                            break
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        text += delta.get("content") or ""
                if not ok:
                    failures.append(f"s{sid} t{t}: stream ended without [DONE]")
                if text != REPLY:
                    failures.append(f"s{sid} t{t}: streamed reply {text[:60]!r}")
            else:
                r = await client.post(f"{base}/v1/chat/completions", json=body, timeout=120)
                if r.status_code != 200:
                    failures.append(f"s{sid} t{t}: status {r.status_code} {r.text[:120]}")
                    return
                d = r.json()
                msg = d["choices"][0]["message"]
                if d.get("object") != "chat.completion" or msg.get("role") != "assistant":
                    failures.append(f"s{sid} t{t}: response shape {list(d)[:6]}")
                if msg.get("content") != REPLY:
                    failures.append(f"s{sid} t{t}: reply {str(msg.get('content'))[:60]!r}")
                if not isinstance(d.get("usage", {}).get("prompt_tokens"), int):
                    failures.append(f"s{sid} t{t}: usage missing")
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
            failures.append(f"s{sid} t{t}: {type(e).__name__}: {e}")
            return
        lat.append(time.perf_counter() - t0)
        history.append({"role": "assistant", "content": REPLY})


HOSTILE = [
    ("not json", b"{this is not json"),
    ("wrong types", json.dumps({"model": 5, "messages": "hello"}).encode()),
    ("no messages", json.dumps({"model": "x"}).encode()),
    ("empty messages", json.dumps({"model": "x", "messages": []}).encode()),
    ("bad role", json.dumps({"model": "x", "messages": [
        {"role": "wizard", "content": "HOSTILE-PROBE hi"}]}).encode()),
    ("content object", json.dumps({"model": "x", "messages": [
        {"role": "user", "content": {"a": "HOSTILE-PROBE"}}]}).encode()),
    # A lone UTF-16 surrogate is legal JSON and cannot be encoded as UTF-8.
    ("lone surrogate", b'{"model": "x", "messages": [{"role": "user", '
                       b'"content": "HOSTILE-PROBE \\ud800 hi"}]}'),
]


async def _hostile(client, base) -> list[str]:
    bad = []
    for name, raw in HOSTILE:
        r = await client.post(f"{base}/v1/chat/completions", content=raw,
                              headers={"Content-Type": "application/json"}, timeout=30)
        if r.status_code >= 500:
            bad.append(f"hostile '{name}': {r.status_code} {r.text[:120]}")
    return bad


async def _run(args) -> dict:
    up = _Upstream()
    port = _free_port()
    storage = tempfile.mkdtemp(prefix="tokenmizer-load-")
    proc = _start_server(up.url, port, args.workers, storage)
    base = f"http://127.0.0.1:{port}"
    rss0 = _rss_mb(proc.pid)
    lat: list[float] = []
    failures: list[str] = []
    try:
        limits = httpx.Limits(max_connections=args.sessions + 8)
        async with httpx.AsyncClient(limits=limits) as client:
            hostile_before = await _hostile(client, base)
            rng = random.Random(args.seed)
            t0 = time.perf_counter()
            await asyncio.gather(*[
                _session(client, base, up, sid, args.turns, args.stream_share, lat, failures,
                         random.Random(rng.random()))
                for sid in range(args.sessions)])
            wall = time.perf_counter() - t0
            hostile_after = await _hostile(client, base)
            alive = (await client.get(f"{base}/health", timeout=10)).status_code < 500
        rss1 = _rss_mb(proc.pid)
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        up.close()
    lat.sort()
    pct = (lambda p: round(lat[min(len(lat) - 1, int(p * len(lat)))] * 1000, 1)) if lat else (lambda p: None)
    return {
        "workers": args.workers, "sessions": args.sessions, "turns_each": args.turns,
        "requests_ok": len(lat), "requests_expected": args.sessions * args.turns,
        "wall_seconds": round(wall, 1), "throughput_rps": round(len(lat) / wall, 1),
        "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "p99": pct(0.99),
                       "mean": round(st.mean(lat) * 1000, 1) if lat else None},
        "server_rss_mb": {"start": round(rss0, 1), "end": round(rss1, 1)},
        "upstream_requests": up.requests,
        "response_failures": failures[:20], "response_failure_count": len(failures),
        "upstream_violations": up.violations[:20], "upstream_violation_count": len(up.violations),
        "hostile_5xx": hostile_before + hostile_after, "alive_after": alive,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sessions", type=int, default=24)
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--stream-share", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    res = asyncio.run(_run(args))
    print(json.dumps(res, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
    clean = (res["response_failure_count"] == 0 and res["upstream_violation_count"] == 0
             and not res["hostile_5xx"] and res["alive_after"]
             and res["requests_ok"] == res["requests_expected"])
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
