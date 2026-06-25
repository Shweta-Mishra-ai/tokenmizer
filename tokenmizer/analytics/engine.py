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


_COST_PER_1K = {
    "claude": 0.003, "anthropic": 0.003,
    "openai": 0.005, "gpt": 0.005,
    "gemini": 0.001,
    "deepseek": 0.0014,
    "mistral": 0.002,
    "default": 0.003,
}


def _cost(tokens: int, provider: str) -> float:
    rate = _COST_PER_1K.get(provider.lower(), _COST_PER_1K["default"])
    return (tokens / 1000) * rate


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

    def __init__(self):
        self._records: List[AnalyticsRecord] = []
        self._by_provider: Dict[str, List[AnalyticsRecord]] = defaultdict(list)
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

    def _period_stats(self, cutoff: float) -> PeriodStats:
        stats = PeriodStats()
        for r in self._records:
            if r.timestamp < cutoff:
                continue
            stats.requests += 1
            stats.tokens_saved += r.tokens_saved
            stats.tokens_used += r.input_tokens_sent + r.output_tokens
            stats.cost_saved += _cost(r.tokens_saved, r.provider)
            stats.cost_actual += _cost(r.input_tokens_sent + r.output_tokens, r.provider)
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
            "total_requests": len(self._records),
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
            "by_provider": {p: len(recs) for p, recs in self._by_provider.items()},
            "suggestions": self.generate_suggestions(),
            # FIXED: persistence failures (checkpoint/graph/redis writes that
            # silently failed) are now visible here instead of only in logs.
            # Non-zero values mean data was lost — investigate immediately.
            "persist_failures": self.persist_failures,
        }
