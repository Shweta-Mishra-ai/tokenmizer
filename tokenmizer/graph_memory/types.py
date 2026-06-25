"""
Core data types for Graph Memory.

Extracted from graph.py to keep that file focused on GraphMemory behavior.
Re-exported from graph.py for backward compatibility — existing imports like
`from tokenmizer.graph_memory.graph import NodeType` continue to work.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class NodeType(str, Enum):
    TASK = "task"
    FILE = "file"
    DECISION = "decision"
    ERROR = "error"
    CONCEPT = "concept"
    DEPENDENCY = "dependency"
    API = "api"
    PROJECT = "project"
    AGENT = "agent"
    # V4 additions
    ENVIRONMENT = "environment"   # runtime env, versions, infra
    GOAL = "goal"                 # top-level session objective
    TEST = "test"                 # test file / test result
    ENDPOINT = "endpoint"         # HTTP endpoint definition
    SCHEMA = "schema"             # data model / DB schema


class NodeStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"       # active — shown in resume (GREEN)
    FAILED = "failed"
    SUPERSEDED = "superseded"     # replaced by newer decision (YELLOW) — kept in history
    INVALIDATED = "invalidated"   # explicitly wrong/cancelled (RED) — kept as warning
    ARCHIVED = "archived"         # old but valid, not relevant now (GRAY)
    MODIFIED = "modified"         # alias for SUPERSEDED — backward compat


class EdgeType(str, Enum):
    DEPENDS_ON = "depends_on"
    RELATED_TO = "related_to"
    IMPLEMENTS = "implements"
    FIXES = "fixes"
    BLOCKS = "blocks"
    PART_OF = "part_of"
    SUPERSEDES = "supersedes"


@dataclass
class MemoryNode:
    id: str
    type: NodeType
    label: str
    status: NodeStatus = NodeStatus.PENDING
    summary: str = ""
    importance: float = 0.5       # 0–1, used in pruning
    confidence: float = 0.7       # 0–1, from GraphValidator
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    valid_from: float = field(default_factory=time.time)   # when this fact became true
    valid_until: float = field(default=0.0)                # 0.0 = currently valid
    _evicted: bool = field(default=False, repr=False)

    def is_valid_at(self, t: float) -> bool:
        """True if this node was active at time t."""
        return self.valid_from <= t and (self.valid_until == 0.0 or self.valid_until > t)

    def touch(self) -> None:
        self.updated_at = time.time()
        self.importance = min(1.0, self.importance + 0.05)

    def age_days(self) -> float:
        return (time.time() - self.updated_at) / 86400


@dataclass
class MemoryEdge:
    source_id: str
    target_id: str
    type: EdgeType
    weight: float = 1.0


@dataclass
class DecisionTransition:
    """
    Full story of why one decision replaced another.

    This is NOT just an edge — it captures the causal chain:
    what triggered the change, why the old decision was wrong,
    what evidence caused the switch, and how confident we are now.

    Stored in a separate SQLite table (not in nodes/edges JSON)
    so it survives graph pruning and is queryable independently.

    Example:
        from_label:      "Use PostgreSQL"
        to_label:        "Use SQLite for MVP"
        trigger:         "cost constraints raised in message 12"
        reason:          "PostgreSQL hosting costs $50/mo, too expensive for MVP"
        evidence:        "User said: 'we have no budget for managed DB right now'"
        confidence_delta: -0.1  (slightly less confident after switch)
    """
    id: str
    session_id: str
    from_decision_id: str
    to_decision_id: str
    from_label: str
    to_label: str
    trigger: str          # what event/message caused the change
    reason: str           # why the old decision was wrong or outdated
    evidence: str         # direct quote or reference from conversation
    confidence_delta: float   # new_confidence - old_confidence (+ = more certain)
    timestamp: float = field(default_factory=time.time)

    def to_context_line(self) -> str:
        """Compact one-line summary for resume context injection."""
        delta_str = (
            f" (+{self.confidence_delta:.0%} confidence)"
            if self.confidence_delta > 0.05
            else f" ({self.confidence_delta:.0%} confidence)"
            if self.confidence_delta < -0.05
            else ""
        )
        return (
            f"Changed: {self.from_label!r} → {self.to_label!r}"
            f"{delta_str}"
            + (f" | Reason: {self.reason[:80]}" if self.reason else "")
        )
