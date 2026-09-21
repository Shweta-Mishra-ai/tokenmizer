"""
Does the resume block still carry what the dropped turns said?

Windowing replaces every turn older than the protected tail with the
graph's context block. Everything the ontology captured survives that, by
construction — it is a node. What does not survive is everything else: a
budget, a deadline, a licence restriction, a performance target. Each is
one sentence, said once, and none of them is a task, a decision, a file
or an error.

This measures that gap directly. For each session it windows the
transcript the way the proxy does, builds the resume block, and asks how
many of the session's out-of-ontology facts are still readable in it.
Reported with and without `graph.record_span_summary`, so the feature is
justified by a number rather than by the argument for it.

HONEST LIMITATION, read this before quoting the number. The fixtures
below were written by the same person who wrote the selector in
`summary.py`, so a high score demonstrates the mechanism works end to end
— it does NOT demonstrate that the selector generalises to phrasings
nobody had in mind. It is the same caveat the extraction eval carries for
its synthetic half, and the same answer applies: the number to trust is
the one from captured transcripts, which is why the real sessions from
`benchmarks/eval/corpus` are scored here too, for token cost and for
regressions in what the graph already held.

Run:  python -m benchmarks.resume_quality.runner
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tokenmizer.compression.window import SmartMessageWindow  # noqa: E402
from tokenmizer.core.tokenizer import count_tokens  # noqa: E402
from tokenmizer.graph_memory.graph import GraphMemory  # noqa: E402

# Each fixture: a session whose opening turns state a fact the ontology
# has no node for, followed by enough ordinary work that those turns are
# windowed away. `expects` lists a distinctive token from each fact — a
# substring match on the resume block, so the score does not depend on
# how the sentence was clipped.
FIXTURES = [
    {
        "id": "bundle_budget",
        "expects": ["500KB", "AGPL"],
        "messages": [
            {"role": "user", "content":
                "New dashboard. Two hard rules: keep the JS bundle under 500KB "
                "gzipped, and we cannot ship anything AGPL licensed — legal "
                "signed off on MIT and Apache only."},
            {"role": "assistant", "content":
                "Understood. Starting with the route scaffolding."},
            {"role": "user", "content": "how is it going"},
            {"role": "assistant", "content":
                "Decided: Vite for the build. Completed: route scaffolding in "
                "src/routes.tsx. Working on the chart components."},
            {"role": "user", "content": "keep going"},
            {"role": "assistant", "content":
                "Completed: chart components in src/components/Chart.tsx. "
                "Decided: tanstack query for the data layer."},
            {"role": "user", "content": "what about the tests"},
            {"role": "assistant", "content":
                "Completed: unit tests in src/components/Chart.test.tsx. "
                "Working on the e2e suite."},
            {"role": "user", "content": "and the build"},
            {"role": "assistant", "content":
                "Completed: CI build step in .github/workflows/ci.yml. "
                "Fixed: flaky snapshot test by pinning the timezone."},
        ],
    },
    {
        "id": "latency_target",
        "expects": ["120ms", "eu-west-1"],
        "messages": [
            {"role": "user", "content":
                "Rewriting the pricing service. The p99 must stay under 120ms, "
                "and everything has to run in eu-west-1 for data residency."},
            {"role": "assistant", "content": "Got it. Looking at the current shape."},
            {"role": "user", "content": "start with the storage"},
            {"role": "assistant", "content":
                "Decided: Postgres with a read replica. Completed: the schema in "
                "migrations/001_pricing.sql."},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content":
                "Completed: the pricing resolver in internal/pricing/resolver.go. "
                "Working on the cache layer."},
            {"role": "user", "content": "cache?"},
            {"role": "assistant", "content":
                "Decided: Redis for the hot path. Completed: cache wiring in "
                "internal/pricing/cache.go."},
            {"role": "user", "content": "how are we looking"},
            {"role": "assistant", "content":
                "Completed: load test harness in tools/loadtest.go. "
                "Fixed: a connection leak in the replica pool."},
        ],
    },
    {
        "id": "deadline_and_owner",
        "expects": ["14th", "on-call"],
        "messages": [
            {"role": "user", "content":
                "Incident follow-up work. This must be done by the 14th — the "
                "customer demo is that morning — and nothing ships without the "
                "on-call engineer approving it."},
            {"role": "assistant", "content": "Understood. Reading the postmortem."},
            {"role": "user", "content": "start on the retries"},
            {"role": "assistant", "content":
                "Decided: exponential backoff with jitter. Completed: the retry "
                "wrapper in internal/http/retry.go."},
            {"role": "user", "content": "then the alerts"},
            {"role": "assistant", "content":
                "Completed: alert rules in ops/alerts.yaml. Working on the runbook."},
            {"role": "user", "content": "keep going"},
            {"role": "assistant", "content":
                "Completed: the runbook in docs/runbooks/pricing.md. "
                "Decided: page on error rate, not on latency."},
            {"role": "user", "content": "status"},
            {"role": "assistant", "content":
                "Completed: dashboards in ops/grafana/pricing.json. "
                "Fixed: a duplicate alert on the staging cluster."},
        ],
    },
]


def _run(messages: list[dict], *, summarise: bool) -> tuple[str, int]:
    """Window the session the way the proxy does; return the resume block
    the next turn would be sent, and the graph's node count."""
    with tempfile.TemporaryDirectory() as d:
        graph = GraphMemory("resume-quality", storage_dir=d)
        graph.extract_from_messages(messages, incremental=False)
        if summarise:
            # Exactly what SmartMessageWindow does, minus the windowing
            # itself: the span it is about to replace.
            window = SmartMessageWindow(token_budget=1, protect_recent=6)
            conv = [m for m in messages if m.get("role") != "system"]
            split = max(0, len(conv) - window.protect_recent)
            while split < len(conv) and conv[split].get("role") != "user":
                split += 1
            graph.record_span_summary(conv[:split])
        return graph.to_context_block(token_budget=400), len(graph._nodes)


def _hits(block: str, expects: list[str]) -> int:
    lowered = block.lower()
    return sum(1 for e in expects if e.lower() in lowered)


def main() -> int:
    print("=" * 74)
    print("TokenMizer — resume quality: what survives windowing")
    print("=" * 74)
    print("Facts stated in turns that windowing drops, and whether the resume")
    print("block still carries them. See this module's docstring for what the")
    print("fixture number does and does not prove.")
    print()

    header = f"  {'session':<22}{'kept':>10}{'kept':>10}{'block':>9}{'block':>9}"
    print(header)
    print(f"  {'':<22}{'(before)':>10}{'(after)':>10}{'(before)':>9}{'(after)':>9}")
    print("  " + "-" * 60)

    want = kept_before = kept_after = 0
    tokens_before = tokens_after = 0

    for fx in FIXTURES:
        before, _ = _run(fx["messages"], summarise=False)
        after, _ = _run(fx["messages"], summarise=True)

        b, a = _hits(before, fx["expects"]), _hits(after, fx["expects"])
        n = len(fx["expects"])
        tb, ta = count_tokens(before), count_tokens(after)

        want += n
        kept_before += b
        kept_after += a
        tokens_before += tb
        tokens_after += ta

        print(f"  {fx['id']:<22}{f'{b}/{n}':>10}{f'{a}/{n}':>10}{tb:>9}{ta:>9}")

    print("  " + "-" * 60)
    pct_b = 100.0 * kept_before / want if want else 0.0
    pct_a = 100.0 * kept_after / want if want else 0.0
    print(f"  {'total':<22}{f'{pct_b:.0f}%':>10}{f'{pct_a:.0f}%':>10}"
          f"{tokens_before:>9}{tokens_after:>9}")
    print()
    cost = tokens_after - tokens_before
    print(f"  Out-of-ontology retention : {pct_b:.0f}% -> {pct_a:.0f}%")
    print(f"  Resume block cost         : +{cost} tokens over {len(FIXTURES)} sessions "
          f"({cost / max(1, len(FIXTURES)):.0f} per session)")
    print()

    # The regression that matters more than the feature: the block is
    # budgeted, so anything added can push out what was already there.
    print("Captured transcripts — the block must not lose what it already held")
    print()
    try:
        from benchmarks.eval import corpus as corpus_mod
    except Exception as e:                          # pragma: no cover
        print(f"  (eval corpus unavailable: {e})")
        return 0

    regressions = 0
    for session in corpus_mod.load(None):
        if session.origin != "real":
            continue
        before, _ = _run(session.messages, summarise=False)
        after, _ = _run(session.messages, summarise=True)
        lost = [line for line in before.splitlines()
                if line.split(":")[0] not in after]
        mark = "ok" if not lost else "LOST"
        if lost:
            regressions += 1
        print(f"  {mark:<5}{session.id:<28}"
              f"{count_tokens(before):>5} -> {count_tokens(after):>5} tokens")
        for line in lost:
            print(f"        dropped section: {line[:60]}")

    print()
    print(f"  Sections lost on real sessions: {regressions}")
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
