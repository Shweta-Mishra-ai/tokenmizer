"""
Latency benchmark for TokenMizer proxy.
Measures p50/p95/p99 latency across 50 requests.

Run: python -m benchmarks.latency.runner
Requires: tokenmizer serve running at localhost:8000
"""
import asyncio
import os
import statistics
import time

import httpx

URL = os.getenv("TOKENMIZER_URL", "http://localhost:8000")
REQUESTS = 50
CONCURRENCY = 5

PROMPTS = [
    "What is a context window?",
    "Explain async/await in Python.",
    "What is a JWT token?",
    "How does Redis work?",
    "Explain REST vs GraphQL.",
]


async def single_request(client: httpx.AsyncClient, prompt: str) -> tuple[float, bool]:
    t0 = time.monotonic()
    r = await client.post(
        f"{URL}/v1/chat/completions",
        json={
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
        },
        timeout=30.0,
    )
    # An error answer is not a latency sample: without this, 50 requests
    # that all came back 502 or 429 printed "50/50 succeeded" with timings.
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
    ms = (time.monotonic() - t0) * 1000
    return ms, bool((r.json().get("tokenmizer") or {}).get("cache_hit"))


async def run():
    print("\nTokenMizer Latency Benchmark")
    print(f"  Target: {URL}")
    print(f"  Requests: {REQUESTS}, Concurrency: {CONCURRENCY}")
    print()

    # Health check
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{URL}/health", timeout=5)
            assert r.status_code == 200
            print("  PASS: Server reachable")
        except Exception as e:
            print(f"  FAIL: Server not reachable: {e}")
            print("     Start with: tokenmizer serve")
            return 1

    latencies = []
    sem = asyncio.Semaphore(CONCURRENCY)

    async def bounded(client, prompt):
        async with sem:
            return await single_request(client, prompt)

    async with httpx.AsyncClient() as client:
        tasks = [
            bounded(client, PROMPTS[i % len(PROMPTS)])
            for i in range(REQUESTS)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    failed = 0
    hits, misses = [], []
    for r in results:
        if isinstance(r, tuple):
            latencies.append(r[0])
            (hits if r[1] else misses).append(r[0])
        else:
            failed += 1
            print(f"  Note:  Request failed: {r}")

    if not latencies:
        print("  FAIL: No successful requests")
        return 1

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)] if len(latencies) >= 100 else latencies[-1]

    print(f"  Results ({len(latencies)}/{REQUESTS} succeeded):")
    print(f"    p50 latency:  {p50:.0f} ms")
    print(f"    p95 latency:  {p95:.0f} ms")
    print(f"    p99 latency:  {p99:.0f} ms")
    print(f"    min / max:    {min(latencies):.0f} / {max(latencies):.0f} ms")
    # The prompts repeat, so once the cache is warm most requests never reach
    # the provider; the overall figures alone read as proxy overhead.
    print(f"    cache hits:   {len(hits)} of {len(latencies)}")
    for label, xs in (("misses", misses), ("hits", hits)):
        if xs:
            print(f"    {label + ' p50:':<14}{statistics.median(xs):.0f} ms")
    print()
    if failed:
        print(f"  FAIL: {failed} of {REQUESTS} requests failed; the figures above are partial")
        return 1
    return 0


if __name__ == "__main__":
    # The exit code is the verdict: a run that could not reach the server
    # or lost requests used to exit 0 and read as a pass.
    raise SystemExit(asyncio.run(run()))
