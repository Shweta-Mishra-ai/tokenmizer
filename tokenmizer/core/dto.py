"""
DTOs — typed data transfer objects for every layer boundary.

Rule: no raw dict crosses a layer boundary.
Each module owns its output DTO. Callers unpack what they need.

tokenmizer/core/dto.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Graph layer ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GraphNodeDTO:
    id: str
    type: str
    label: str
    status: str
    summary: str
    importance: float
    confidence: float
    age_days: float


@dataclass(frozen=True)
class GraphEdgeDTO:
    source_id: str
    target_id: str
    type: str
    weight: float


@dataclass(frozen=True)
class GraphStatsDTO:
    session_id: str
    node_count: int
    edge_count: int
    by_type: dict
    by_status: dict
    processed_messages: int
    avg_confidence: float


# ── Checkpoint layer ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CheckpointSummaryDTO:
    """Lightweight — for list endpoints."""
    checkpoint_id: str
    session_id: str
    created_at: float
    context_pct: float
    trigger: str
    message_count: int
    resume_tokens: int


@dataclass(frozen=True)
class ResumeDTO:
    session_id: str
    checkpoint_id: str
    level: str
    resume_context: str
    token_count: int
    node_count: int


# ── Provider layer ────────────────────────────────────────────────────────────

@dataclass
class LLMResponseDTO:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    cached: bool = False
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ── Compression layer ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CompressionResultDTO:
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    quality_score: float          # 0–1; if < threshold, original was used
    strategy_used: str
    was_compressed: bool


@dataclass(frozen=True)
class OutputTrimResultDTO:
    original_tokens: int
    trimmed_tokens: int
    tokens_saved: int
    text: str


# ── Cache layer ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CacheStatsDTO:
    entries: int
    max_size: int
    utilization_pct: float
    evictions: int
    hit_rate: float
    hit_exact: int
    hit_semantic: int
    miss: int
    semantic_available: bool


# ── File intelligence layer ───────────────────────────────────────────────────

@dataclass(frozen=True)
class FileExtractionDTO:
    file_type: str
    filename: str
    original_size_bytes: int
    original_tokens: int
    extracted_tokens: int
    tokens_saved: int
    savings_pct: float
    content: str
    summary: str
    strategy_used: str
    was_truncated: bool


# ── Analytics layer ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PeriodStatsDTO:
    period: str                    # "daily" | "weekly" | "monthly"
    requests: int
    tokens_saved: int
    savings_pct: float
    cost_saved_usd: float


@dataclass(frozen=True)
class AnalyticsSummaryDTO:
    total_requests: int
    daily: PeriodStatsDTO
    weekly: PeriodStatsDTO
    monthly: PeriodStatsDTO
    layer_breakdown: dict
    by_provider: dict
    suggestions: list[str]


# ── Chat API layer ─────────────────────────────────────────────────────────────

@dataclass
class ChatSavingsDTO:
    file_extraction: int = 0
    compression: int = 0
    output_trim: int = 0
    cache: int = 0
    windowing: int = 0
    routing: int = 0

    @property
    def total(self) -> int:
        return (self.file_extraction + self.compression + self.output_trim
                + self.cache + self.windowing + self.routing)

    def to_dict(self) -> dict:
        return {
            "file_extraction": self.file_extraction,
            "compression": self.compression,
            "output_trim": self.output_trim,
            "cache": self.cache,
            "windowing": self.windowing,
            "routing": self.routing,
        }
