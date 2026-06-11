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


async def single_request(client: httpx.AsyncClient, prompt: str) -> float:
    t0 = time.monotonic()
    await client.post(
        f"{URL}/v1/chat/completions",
        json={
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
        },
        timeout=30.0,
    )
    return (time.monotonic() - t0) * 1000  # ms


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
            print("  ✅ Server reachable")
        except Exception as e:
            print(f"  ❌ Server not reachable: {e}")
            print("     Start with: tokenmizer serve")
            return

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

    for r in results:
        if isinstance(r, float):
            latencies.append(r)
        else:
            print(f"  ⚠️  Request failed: {r}")

    if not latencies:
        print("  ❌ No successful requests")
        return

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)] if len(latencies) >= 100 else latencies[-1]

    print(f"  Results ({len(latencies)}/{REQUESTS} succeeded):")
    print(f"    p50 latency:  {p50:.0f} ms")
    print(f"    p95 latency:  {p95:.0f} ms")
    print(f"    p99 latency:  {p99:.0f} ms")
    print(f"    min / max:    {min(latencies):.0f} / {max(latencies):.0f} ms")
    print()


if __name__ == "__main__":
    asyncio.run(run())
