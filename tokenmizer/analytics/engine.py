"""Analytics engine — daily/weekly/monthly rollups."""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PeriodStats:
    requests: int = 0
    tokens_saved: int = 0
    tokens_used: int = 0
    cost_saved: float = 0.0
    cost_actual: float = 0.0

    @property
    def savings_pct(self) -> float:
        total = self.tokens_saved + self.tokens_used
        return (self.tokens_saved / total * 100) if total > 0 else 0.0


# Per-1K-token rates as (input, output).
#
# These used to be a SINGLE blended rate per provider, applied to input
# and output tokens alike. Output tokens cost several times more than
# input on every provider here (5x on Anthropic, 4x on OpenAI), so both
# `cost_saved_usd` and `cost_actual` were materially wrong — and wrong in
# a flattering direction for the savings figure the dashboard leads with.
# Splitting them is the difference between a number you can put in front
# of a finance team and a number you can't.
#
# Still approximate: real pricing is per-MODEL, not per-provider, and
# changes over time. Treat these as an estimate, which is why the field
# is named cost_saved_usd and not billed_amount.
_COST_PER_1K = {
    "claude":    (0.003,  0.015),
    "anthropic": (0.003,  0.015),
    "openai":    (0.005,  0.020),
    "gpt":       (0.005,  0.020),
    "gemini":    (0.001,  0.004),
    "deepseek":  (0.0014, 0.0028),
    "mistral":   (0.002,  0.006),
    "default":   (0.003,  0.015),
}


def _cost(input_tokens: int, output_tokens: int, provider: str) -> float:
    """Estimated USD cost, charging input and output at their own rates."""
    in_rate, out_rate = _COST_PER_1K.get(provider.lower(), _COST_PER_1K["default"])
    return (input_tokens / 1000) * in_rate + (output_tokens / 1000) * out_rate


@dataclass
class AnalyticsRecord:
    timestamp: float
    session_id: str
    provider: str
    model: str
    input_tokens_original: int
    input_tokens_sent: int
    output_tokens: int
    tokens_saved: int
    latency_ms: float
    cache_hit: bool
    layer_savings: Dict[str, int] = field(default_factory=dict)


class AnalyticsEngine:

    # Hard cap on retained per-request records.
    #
    # This list was unbounded and appended to on EVERY request, with a
    # second copy of each reference in _by_provider — the one structure
    # in the codebase without a cap, while graph cache, session locks,
    # semantic cache and rate-limiter buckets are all LRU-bounded. A
    # long-running proxy grew it until the process died. summary() also
    # scans the whole list five times (daily/weekly/monthly +
    # layer_breakdown + generate_suggestions re-deriving daily), so
    # /api/stats got linearly slower for the life of the process too.
    #
    # The monthly rollup is the longest window anything reads, so records
    # older than that are dead weight regardless of count. Both bounds
    # are enforced: age first, then a hard ceiling.
    MAX_RECORDS = 50_000
    MAX_RECORD_AGE_SECONDS = 31 * 86_400

    def __init__(self, max_records: int = MAX_RECORDS):
        self._max_records = max_records
        self._records: List[AnalyticsRecord] = []
        self._by_provider: Dict[str, List[AnalyticsRecord]] = defaultdict(list)
        # Provider counts must survive record trimming — a total that
        # silently decreases as old records age out is worse than no
        # total at all.
        self._provider_totals: Dict[str, int] = defaultdict(int)
        self._total_requests = 0
        # FIXED: previously, silent failures (checkpoint save, graph
        # eviction persist, Redis write, AND background LLM extraction
        # errors) were caught, logged at low severity, and otherwise
        # invisible — no way to know in production whether data loss or
        # feature degradation was happening without grepping logs. This
        # counter makes "how many times did something silently fail this
        # session" a queryable number via /api/stats instead of a fact
        # buried in a log line. Dict key is `persist_failures` for API
        # stability even though it now covers a slightly broader category
        # than literal persistence (see record_silent_failure docstring).
        self._persist_failures: Dict[str, int] = defaultdict(int)

    def record_silent_failure(self, source: str) -> None:
        """Track a failure that would otherwise be invisible outside debug
        logs — persistence (checkpoint save, graph eviction, Redis write)
        AND non-persistence failures like background LLM extraction
        errors. The common thread: all of these used to fail silently
        with zero visibility outside of logs nobody watches by default.
        Call this from every place that catches such an exception — it
        costs one dict increment and turns 'silent forever' into 'visible
        in /api/stats'."""
        self._persist_failures[source] += 1

    @property
    def persist_failures(self) -> Dict[str, int]:
        return dict(self._persist_failures)

    def record(
        self,
        session_id: str,
        provider: str,
        model: str,
        input_tokens_original: int,
        input_tokens_sent: int,
        output_tokens: int,
        tokens_saved: int,
        latency_ms: float,
        cache_hit: bool,
        layer_savings: Dict[str, int] | None = None,
    ) -> None:
        r = AnalyticsRecord(
            timestamp=time.time(),
            session_id=session_id,
            provider=provider,
            model=model,
            input_tokens_original=input_tokens_original,
            input_tokens_sent=input_tokens_sent,
            output_tokens=output_tokens,
            tokens_saved=tokens_saved,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            layer_savings=layer_savings or {},
        )
        self._records.append(r)
        self._by_provider[provider].append(r)
        self._provider_totals[provider] += 1
        self._total_requests += 1
        self._trim()

    def _trim(self) -> None:
        """Drop records older than the longest reporting window, then
        enforce the hard ceiling. Records are appended in timestamp
        order, so the oldest are always at the front."""
        cutoff = time.time() - self.MAX_RECORD_AGE_SECONDS
        drop = 0
        for r in self._records:
            if r.timestamp >= cutoff:
                break
            drop += 1
        if len(self._records) - drop > self._max_records:
            drop = len(self._records) - self._max_records
        if drop <= 0:
            return
        dropped = self._records[:drop]
        del self._records[:drop]
        # Keep the per-provider index consistent with the trimmed list,
        # or it becomes the unbounded leak this cap was meant to remove.
        stale = {id(r) for r in dropped}
        for prov, recs in list(self._by_provider.items()):
            kept = [r for r in recs if id(r) not in stale]
            if kept:
                self._by_provider[prov] = kept
            else:
                del self._by_provider[prov]

    def _period_stats(self, cutoff: float) -> PeriodStats:
        stats = PeriodStats()
        for r in self._records:
            if r.timestamp < cutoff:
                continue
            stats.requests += 1
            stats.tokens_saved += r.tokens_saved
            stats.tokens_used += r.input_tokens_sent + r.output_tokens
            # Savings are input-side (compression, windowing, file
            # intelligence and cache all reduce the prompt), so they are
            # priced at the input rate rather than a blend that silently
            # inflated them with output pricing.
            stats.cost_saved += _cost(r.tokens_saved, 0, r.provider)
            stats.cost_actual += _cost(r.input_tokens_sent, r.output_tokens, r.provider)
        return stats

    @property
    def daily(self) -> PeriodStats:
        return self._period_stats(time.time() - 86_400)

    @property
    def weekly(self) -> PeriodStats:
        return self._period_stats(time.time() - 7 * 86_400)

    @property
    def monthly(self) -> PeriodStats:
        return self._period_stats(time.time() - 30 * 86_400)

    def layer_breakdown(self) -> Dict[str, int]:
        totals: Dict[str, int] = defaultdict(int)
        for r in self._records:
            for layer, saved in r.layer_savings.items():
                totals[layer] += saved
        return dict(totals)

    def generate_suggestions(self) -> list[str]:
        suggestions = []
        d = self.daily
        if d.requests == 0:
            suggestions.append("No requests yet — start sending requests to see analytics.")
            return suggestions
        if d.tokens_saved == 0:
            suggestions.append("Enable compression in tokenmizer.yaml to start saving tokens.")
        breakdown = self.layer_breakdown()
        if breakdown.get("cache", 0) == 0:
            suggestions.append("Semantic cache has no hits yet — similar queries will be cached automatically.")
        return suggestions

    def summary(self) -> dict:
        d, w, m = self.daily, self.weekly, self.monthly
        return {
            # Lifetime count, not len(self._records) — that would silently
            # shrink as old records are trimmed.
            "total_requests": self._total_requests,
            "retained_records": len(self._records),
            "daily": {
                "requests": d.requests,
                "tokens_saved": d.tokens_saved,
                "savings_pct": round(d.savings_pct, 1),
                "cost_saved_usd": round(d.cost_saved, 4),
            },
            "weekly": {
                "requests": w.requests,
                "tokens_saved": w.tokens_saved,
                "savings_pct": round(w.savings_pct, 1),
                "cost_saved_usd": round(w.cost_saved, 4),
            },
            "monthly": {
                "requests": m.requests,
                "tokens_saved": m.tokens_saved,
                "savings_pct": round(m.savings_pct, 1),
                "cost_saved_usd": round(m.cost_saved, 4),
            },
            "layer_breakdown": self.layer_breakdown(),
            "by_provider": dict(self._provider_totals),
            "suggestions": self.generate_suggestions(),
            # FIXED: persistence failures (checkpoint/graph/redis writes that
            # silently failed) are now visible here instead of only in logs.
            # Non-zero values mean data was lost — investigate immediately.
            "persist_failures": self.persist_failures,
        }
