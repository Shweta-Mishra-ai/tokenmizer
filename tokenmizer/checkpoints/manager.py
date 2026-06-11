"""
Checkpoint Manager — creates and restores session checkpoints.

Key fixes over V3:
- Extracts from FULL message history, not just last 10
- Tiered resume blocks (critical / standard / full)
- Graph diff between checkpoints
- Accurate token counting via tiktoken
- SQLite persistence
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tokenmizer.graph_memory.graph import GraphMemory
from tokenmizer.core.tokenizer import count_tokens, count_messages_tokens

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    checkpoint_id: str
    session_id: str
    created_at: float
    context_pct: float
    trigger: str                   # "auto_threshold" | "manual" | "provider_switch"
    message_count: int
    graph_snapshot: dict           # full graph state at checkpoint time
    graph_diff: dict               # diff from previous checkpoint
    resume_critical: str           # ~100 tokens — must-know facts
    resume_standard: str           # ~300 tokens — normal resume
    resume_full: str               # ~600 tokens — deep resume
    model: str = ""
    next_action: str = ""

    @property
    def resume_tokens(self) -> int:
        return count_tokens(self.resume_standard)


class CheckpointManager:
    """
    Manages checkpoints for all sessions.
    Stored in SQLite — survives restarts.
    """

    def __init__(self, storage_dir: str = "./checkpoints"):
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._dir / "checkpoints.db"
        self._init_db()
        self._prev_snapshots: dict[str, dict] = {}  # session_id → last snapshot

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    context_pct REAL,
                    trigger TEXT,
                    message_count INTEGER,
                    data_json TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ckpt_session ON checkpoints(session_id)")
            conn.commit()

    def create(
        self,
        session_id: str,
        messages: list[dict],      # FULL message history — not just recent
        graph: GraphMemory,
        context_pct: float,
        trigger: str = "auto_threshold",
        model: str = "",
    ) -> Checkpoint:
        """Create a checkpoint from the current session state."""
        checkpoint_id = f"ckpt_{uuid.uuid4().hex[:12]}"

        # Force full extraction from ALL messages before snapshotting
        graph.extract_from_messages(messages, incremental=True)

        graph_snapshot = {
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type.value,
                    "label": n.label,
                    "status": n.status.value,
                    "summary": n.summary,
                    "importance": n.importance,
                }
                for n in graph._nodes.values()
                if not n._evicted
            ],
            "edge_count": len(graph._edges),
        }

        # Compute diff from previous checkpoint
        prev = self._prev_snapshots.get(session_id, {"nodes": []})
        graph_diff = self._compute_diff(prev, graph_snapshot)
        self._prev_snapshots[session_id] = graph_snapshot

        # Get last user message as next_action hint
        next_action = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                next_action = msg.get("content", "")[:200]
                break

        # Build tiered resume blocks
        resume_critical = self._build_critical(graph, next_action)
        resume_standard = self._build_standard(graph, next_action)
        resume_full = self._build_full(graph, messages, next_action)

        ckpt = Checkpoint(
            checkpoint_id=checkpoint_id,
            session_id=session_id,
            created_at=time.time(),
            context_pct=context_pct,
            trigger=trigger,
            message_count=len(messages),
            graph_snapshot=graph_snapshot,
            graph_diff=graph_diff,
            resume_critical=resume_critical,
            resume_standard=resume_standard,
            resume_full=resume_full,
            model=model,
            next_action=next_action,
        )

        self._save(ckpt)
        logger.info(
            f"Checkpoint {checkpoint_id}: session={session_id} "
            f"msgs={len(messages)} nodes={len(graph._nodes)} "
            f"context={context_pct:.0%} resume_tokens={ckpt.resume_tokens}"
        )
        return ckpt

    def _build_critical(self, graph: GraphMemory, next_action: str) -> str:
        """~100 tokens. Only open blockers + critical decisions."""
        from tokenmizer.graph_memory.graph import NodeType, NodeStatus
        lines = []

        open_errors = [n for n in graph._nodes.values()
                       if n.type == NodeType.ERROR and n.status == NodeStatus.FAILED]
        if open_errors:
            lines.append("OPEN BUGS: " + " | ".join(e.label for e in open_errors[:3]))

        high_priority_tasks = [n for n in graph._nodes.values()
                               if n.type == NodeType.TASK
                               and n.status == NodeStatus.IN_PROGRESS
                               and n.importance >= 0.8]
        if high_priority_tasks:
            lines.append("CRITICAL WIP: " + " | ".join(t.label for t in high_priority_tasks[:3]))

        decisions = sorted(
            [n for n in graph._nodes.values() if n.type == NodeType.DECISION],
            key=lambda x: x.importance, reverse=True
        )
        if decisions:
            lines.append("KEY DECISIONS: " + " | ".join(d.label for d in decisions[:3]))

        if next_action:
            lines.append(f"LAST REQUEST: {next_action[:100]}")

        return "\n".join(lines)

    def _build_standard(self, graph: GraphMemory, next_action: str) -> str:
        """~300 tokens. Normal resume — goals, tasks, decisions, files."""
        block = graph.to_context_block(token_budget=300)
        if next_action:
            block += f"\nContinue from: {next_action[:150]}"
        return block

    def _build_full(self, graph: GraphMemory, messages: list[dict], next_action: str) -> str:
        """~600 tokens. Deep resume with environment, schemas, dependencies."""
        from tokenmizer.graph_memory.graph import NodeType, NodeStatus
        parts = [self._build_standard(graph, "")]

        env_nodes = [n for n in graph._nodes.values() if n.type == NodeType.ENVIRONMENT]
        if env_nodes:
            parts.append("Environment: " + ", ".join(e.label for e in env_nodes[:8]))

        dep_nodes = [n for n in graph._nodes.values() if n.type == NodeType.DEPENDENCY]
        if dep_nodes:
            parts.append("Dependencies: " + ", ".join(d.label for d in dep_nodes[:10]))

        schema_nodes = [n for n in graph._nodes.values() if n.type == NodeType.SCHEMA]
        if schema_nodes:
            parts.append("Schemas: " + " | ".join(s.label for s in schema_nodes[:4]))

        endpoint_nodes = [n for n in graph._nodes.values() if n.type == NodeType.ENDPOINT]
        if endpoint_nodes:
            parts.append("Endpoints: " + ", ".join(e.label for e in endpoint_nodes[:8]))

        done_tasks = [n for n in graph._nodes.values()
                      if n.type == NodeType.TASK and n.status == NodeStatus.COMPLETED]
        done_tasks.sort(key=lambda x: x.updated_at, reverse=True)
        if done_tasks:
            parts.append("Recently completed: " + " | ".join(t.label for t in done_tasks[:6]))

        if next_action:
            parts.append(f"Continue from: {next_action[:150]}")

        return "\n".join(parts)

    def _compute_diff(self, prev: dict, current: dict) -> dict:
        prev_nodes = {n["id"]: n for n in prev.get("nodes", [])}
        curr_nodes = {n["id"]: n for n in current.get("nodes", [])}
        return {
            "added": [n for nid, n in curr_nodes.items() if nid not in prev_nodes],
            "removed": [n for nid, n in prev_nodes.items() if nid not in curr_nodes],
            "status_changed": [
                {"id": nid, "from": prev_nodes[nid]["status"], "to": curr_nodes[nid]["status"]}
                for nid in curr_nodes
                if nid in prev_nodes
                and curr_nodes[nid]["status"] != prev_nodes[nid]["status"]
            ],
        }

    def _save(self, ckpt: Checkpoint) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO checkpoints
                       (checkpoint_id, session_id, created_at, context_pct, trigger, message_count, data_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        ckpt.checkpoint_id,
                        ckpt.session_id,
                        ckpt.created_at,
                        ckpt.context_pct,
                        ckpt.trigger,
                        ckpt.message_count,
                        json.dumps({
                            "graph_snapshot": ckpt.graph_snapshot,
                            "graph_diff": ckpt.graph_diff,
                            "resume_critical": ckpt.resume_critical,
                            "resume_standard": ckpt.resume_standard,
                            "resume_full": ckpt.resume_full,
                            "model": ckpt.model,
                            "next_action": ckpt.next_action,
                        }),
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")

    def get_latest(self, session_id: str) -> Optional[Checkpoint]:
        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    """SELECT checkpoint_id, session_id, created_at, context_pct,
                              trigger, message_count, data_json
                       FROM checkpoints WHERE session_id=?
                       ORDER BY created_at DESC LIMIT 1""",
                    (session_id,),
                ).fetchone()
            if not row:
                return None
            return self._row_to_checkpoint(row)
        except Exception as e:
            logger.error(f"Checkpoint load failed: {e}")
            return None

    def list_checkpoints(self, session_id: str) -> list[dict]:
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    """SELECT checkpoint_id, created_at, context_pct, trigger, message_count
                       FROM checkpoints WHERE session_id=? ORDER BY created_at DESC""",
                    (session_id,),
                ).fetchall()
            return [
                {"checkpoint_id": r[0], "created_at": r[1], "context_pct": r[2],
                 "trigger": r[3], "message_count": r[4]}
                for r in rows
            ]
        except Exception:
            return []

    def _row_to_checkpoint(self, row) -> Checkpoint:
        data = json.loads(row[6])
        return Checkpoint(
            checkpoint_id=row[0],
            session_id=row[1],
            created_at=row[2],
            context_pct=row[3] or 0.0,
            trigger=row[4] or "unknown",
            message_count=row[5] or 0,
            graph_snapshot=data.get("graph_snapshot", {}),
            graph_diff=data.get("graph_diff", {}),
            resume_critical=data.get("resume_critical", ""),
            resume_standard=data.get("resume_standard", ""),
            resume_full=data.get("resume_full", ""),
            model=data.get("model", ""),
            next_action=data.get("next_action", ""),
        )
