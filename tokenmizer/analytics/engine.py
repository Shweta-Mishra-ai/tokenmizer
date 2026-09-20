"""Analytics engine — daily/weekly/monthly rollups.

Durability: graph state has WAL-mode SQLite, a periodic flusher, corruption
quarantine and a shutdown drain. Analytics had none of it. Every record lived
in a Python list and nothing was ever written to disk, so daily(), weekly()
and monthly() reset to zero on any restart, redeploy or container recycle. A
weekly figure that cannot survive a week is not a weekly figure, and `stats`
is one of the first things a new user tries.

Writes are buffered and flushed on a timer rather than written per request:
record() is called synchronously from the chat handler, and an fsync per
request would put disk latency on the hot path. That gives analytics the same
guarantee the graph already documents — a hard kill loses at most one flush
interval — instead of losing everything.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


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
# These MUST stay split by direction. Output tokens cost several times
# more than input on every provider here (5x on Anthropic, 4x on
# OpenAI), so a single blended rate makes both `cost_saved_usd` and
# `cost_actual` materially wrong — and wrong in a flattering direction
# for the savings figure the dashboard leads with.
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

    def __init__(self, max_records: int = MAX_RECORDS, storage_dir: str | None = None):
        self._max_records = max_records
        # Its own file, not the graph's. graph_memory.db is under a
        # cross-process write lock held for every node/edge save; putting
        # per-request analytics rows behind that lock would make the two
        # contend for no reason, and a corrupt analytics file must never be
        # able to take session memory with it.
        self._db_path = Path(storage_dir) / "analytics.db" if storage_dir else None
        # Rows written but not yet flushed. Bounded by flush frequency, not
        # by request volume — flush() is called on the proxy's existing
        # periodic timer and at shutdown.
        self._pending: List[AnalyticsRecord] = []
        self._records: List[AnalyticsRecord] = []
        self._by_provider: Dict[str, List[AnalyticsRecord]] = defaultdict(list)
        # Provider counts must survive record trimming — a total that
        # silently decreases as old records age out is worse than no
        # total at all.
        self._provider_totals: Dict[str, int] = defaultdict(int)
        self._total_requests = 0
        # Silent failures (checkpoint save, graph
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
        # Work dropped on purpose to stay inside a limit — see record_shed.
        self._shed: Dict[str, int] = defaultdict(int)

        if self._db_path is not None:
            self._init_db()
            self._load()

    # ── Durability ───────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=5.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            conn.close()
            raise
        return conn

    def _init_db(self) -> None:
        """Create the store, or disable persistence and keep serving.

        Analytics are reporting, not correctness. A broken analytics file
        must degrade to the previous in-memory behaviour rather than fail a
        request, so every path here is best-effort and says so in the log.
        """
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS requests (
                        timestamp REAL NOT NULL,
                        session_id TEXT, provider TEXT, model TEXT,
                        input_tokens_original INTEGER, input_tokens_sent INTEGER,
                        output_tokens INTEGER, tokens_saved INTEGER,
                        latency_ms REAL, cache_hit INTEGER, layer_savings TEXT
                    )""")
                # Every read is a time window (daily/weekly/monthly) and
                # trimming deletes by age, so this is the only index needed.
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(timestamp)")
                # Provider totals and the request count must survive record
                # trimming — a lifetime total that silently decreases as old
                # rows age out is worse than no total at all.
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS counters (
                        key TEXT PRIMARY KEY, value INTEGER NOT NULL
                    )""")
        except Exception as e:
            logger.error(
                "Analytics persistence disabled (%s): %s — stats will reset on "
                "restart, requests are unaffected", self._db_path, e)
            self._db_path = None

    def _load(self) -> None:
        """Restore the retained window and the lifetime counters."""
        cutoff = time.time() - self.MAX_RECORD_AGE_SECONDS
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT timestamp, session_id, provider, model, "
                    "input_tokens_original, input_tokens_sent, output_tokens, "
                    "tokens_saved, latency_ms, cache_hit, layer_savings "
                    "FROM requests WHERE timestamp >= ? ORDER BY timestamp "
                    "LIMIT ?", (cutoff, self._max_records)).fetchall()
                counters = dict(conn.execute(
                    "SELECT key, value FROM counters").fetchall())
        except Exception as e:
            logger.error("Could not read analytics history: %s", e)
            return

        for row in rows:
            try:
                layer_savings = json.loads(row[10]) if row[10] else {}
            except (TypeError, ValueError):
                layer_savings = {}
            record = AnalyticsRecord(
                timestamp=row[0], session_id=row[1] or "", provider=row[2] or "",
                model=row[3] or "", input_tokens_original=row[4] or 0,
                input_tokens_sent=row[5] or 0, output_tokens=row[6] or 0,
                tokens_saved=row[7] or 0, latency_ms=row[8] or 0.0,
                cache_hit=bool(row[9]), layer_savings=layer_savings,
            )
            self._records.append(record)
            self._by_provider[record.provider].append(record)

        self._total_requests = int(counters.get("total_requests", 0))
        for key, value in counters.items():
            if key.startswith("provider:"):
                self._provider_totals[key[len("provider:"):]] = int(value)
        if rows:
            logger.info("Analytics: restored %d records", len(rows))

    def flush(self) -> bool:
        """Write buffered records and the lifetime counters. Returns success.

        Called from the proxy's periodic flusher and at shutdown, alongside
        the graph flush, so both have the same worst-case exposure.
        """
        if self._db_path is None or not self._pending:
            return True
        batch, self._pending = self._pending, []
        try:
            with self._connect() as conn:
                conn.executemany(
                    "INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    [(r.timestamp, r.session_id, r.provider, r.model,
                      r.input_tokens_original, r.input_tokens_sent,
                      r.output_tokens, r.tokens_saved, r.latency_ms,
                      int(r.cache_hit), json.dumps(r.layer_savings))
                     for r in batch])
                conn.executemany(
                    "INSERT OR REPLACE INTO counters VALUES (?,?)",
                    [("total_requests", self._total_requests)]
                    + [(f"provider:{p}", n) for p, n in self._provider_totals.items()])
                conn.execute("DELETE FROM requests WHERE timestamp < ?",
                             (time.time() - self.MAX_RECORD_AGE_SECONDS,))
            return True
        except Exception as e:
            # Put the batch back so the next flush retries it rather than
            # dropping the window silently.
            self._pending = batch + self._pending
            logger.error("Analytics flush failed (%d records pending): %s",
                         len(self._pending), e)
            return False

    def record_silent_failure(self, source: str) -> None:
        """Track a failure that would otherwise be invisible outside debug
        logs — persistence (checkpoint save, graph eviction, Redis write)
        AND non-persistence failures like background LLM extraction
        errors. The common thread: all of them can otherwise fail with
        zero visibility outside logs nobody watches by default.
        Call this from every place that catches such an exception — it
        costs one dict increment and turns 'silent forever' into 'visible
        in /api/stats'."""
        self._persist_failures[source] += 1

    @property
    def persist_failures(self) -> Dict[str, int]:
        return dict(self._persist_failures)

    def record_shed(self, source: str) -> None:
        """Track work deliberately dropped to stay inside a limit.

        Distinct from `record_silent_failure` on purpose: a shed is not a
        fault, it is the system doing what it was configured to do, and an
        operator reading /api/stats needs to tell "my cheap provider is
        down" apart from "my proxy is at capacity". They lead to opposite
        actions — fix a key, or raise a limit — and one dict of mixed
        counts cannot say which."""
        self._shed[source] += 1

    @property
    def shed(self) -> Dict[str, int]:
        return dict(self._shed)

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
        if self._db_path is not None:
            self._pending.append(r)
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
            # persistence failures (checkpoint/graph/redis writes that
            # silently failed) are now visible here instead of only in logs.
            # Non-zero values mean data was lost — investigate immediately.
            "persist_failures": self.persist_failures,
            "shed": self.shed,
        }
