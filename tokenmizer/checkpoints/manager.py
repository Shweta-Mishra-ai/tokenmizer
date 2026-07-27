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
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import sqlite3

from tokenmizer.core.errors import CheckpointPersistError
from tokenmizer.core.tokenizer import count_tokens
from tokenmizer.graph_memory.graph import GraphMemory

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

    # Default cap on AUTO-triggered checkpoints retained per session. Each
    # checkpoint snapshots the full graph as JSON (see issue #27), so an
    # unbounded table grows without limit for any long-lived session that
    # legitimately stays near the auto-checkpoint threshold. Manual
    # checkpoints (an explicit user/CLI action) are never subject to this
    # cap — only ones the auto-threshold trigger itself created.
    DEFAULT_AUTO_RETENTION = 20

    def __init__(self, storage_dir: str = "./checkpoints", auto_retention: int = DEFAULT_AUTO_RETENTION):
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._dir / "checkpoints.db"
        self._auto_retention = auto_retention
        self._safe_init_db()
        self._prev_snapshots: dict[str, dict] = {}  # session_id → last snapshot

    def _safe_init_db(self) -> None:
        """Initialize DB, deleting corrupt file if necessary."""
        try:
            self._init_db()
        except Exception:
            logger.warning(f"Checkpoint DB corrupt or unreadable — recreating: {self._db_path}")
            try:
                self._db_path.unlink(missing_ok=True)
            except Exception as del_err:
                logger.error(f"Could not delete corrupt checkpoint DB: {del_err}")
            try:
                self._init_db()
            except Exception as e:
                logger.error(f"Cannot initialize checkpoint DB after cleanup: {e}")

    def _db_connect(self) -> sqlite3.Connection:
        """SQLite connection with WAL mode and timeout for concurrent safety."""
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(str(self._db_path), timeout=5.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        conn = self._db_connect()
        try:
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
        finally:
            conn.close()

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
        if trigger == "auto_threshold":
            self._prune_auto_checkpoints(session_id)
        logger.info(
            f"Checkpoint {checkpoint_id}: session={session_id} "
            f"msgs={len(messages)} nodes={len(graph._nodes)} "
            f"context={context_pct:.0%} resume_tokens={ckpt.resume_tokens}"
        )
        return ckpt

    def _prune_auto_checkpoints(self, session_id: str) -> None:
        """
        Cap AUTO-triggered checkpoints at self._auto_retention for this
        session, deleting the oldest first. Never touches rows with any
        other trigger value (manual, provider_switch, etc.) — those are
        explicit actions and must survive regardless of how many
        auto-threshold checkpoints follow them.

        Best-effort: a failure here must not fail the checkpoint that was
        just successfully saved — worst case the table grows one row
        larger than intended until the next successful prune.
        """
        try:
            conn = self._db_connect()
            try:
                rows = conn.execute(
                    """SELECT checkpoint_id FROM checkpoints
                       WHERE session_id=? AND trigger='auto_threshold'
                       ORDER BY created_at DESC""",
                    (session_id,),
                ).fetchall()
                stale_ids = [r[0] for r in rows[self._auto_retention:]]
                if stale_ids:
                    conn.executemany(
                        "DELETE FROM checkpoints WHERE checkpoint_id=?",
                        [(cid,) for cid in stale_ids],
                    )
                    conn.commit()
                    logger.debug(
                        f"Pruned {len(stale_ids)} old auto-checkpoint(s) for "
                        f"session {session_id} (retention={self._auto_retention})"
                    )
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"Auto-checkpoint retention prune failed for "
                           f"session {session_id} (non-fatal): {e}")

    def _build_critical(self, graph: GraphMemory, next_action: str) -> str:
        """~100 tokens. Only open blockers + critical decisions."""
        from tokenmizer.graph_memory.graph import NodeStatus, NodeType
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
        from tokenmizer.graph_memory.graph import NodeStatus, NodeType
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
        """
        Persist a checkpoint to SQLite.

        FIXED: previously this caught Exception, logged it, and returned
        None — silently. The caller (create()) had no way to know the
        save failed, so callers (including the auto-checkpoint trigger and
        the manual /api/checkpoint endpoint) would report a checkpoint as
        successfully created when nothing was actually written to disk.
        For a tool whose entire pitch is "never lose context," silently
        losing the checkpoint on save failure is the worst possible
        failure mode — the user trusts the safety net fired and finds out
        otherwise only when they try to resume and there's nothing there.

        Now raises CheckpointPersistError so callers can decide how to
        handle it (the API layer already wraps checkpoint creation in
        try/except and returns a proper 500 — this just makes that path
        reachable instead of dead code).
        """
        try:
            conn = self._db_connect()
            try:
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
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Checkpoint save failed for {ckpt.checkpoint_id}: {e}")
            raise CheckpointPersistError(
                f"Failed to persist checkpoint {ckpt.checkpoint_id} for "
                f"session {ckpt.session_id}: {e}"
            ) from e

    def get_latest(self, session_id: str) -> Optional[Checkpoint]:
        try:
            conn = self._db_connect()
            try:
                row = conn.execute(
                    """SELECT checkpoint_id, session_id, created_at, context_pct,
                              trigger, message_count, data_json
                       FROM checkpoints WHERE session_id=?
                       ORDER BY created_at DESC LIMIT 1""",
                    (session_id,),
                ).fetchone()
            finally:
                conn.close()
            if not row:
                return None
            return self._row_to_checkpoint(row)
        except Exception as e:
            logger.error(f"Checkpoint load failed: {e}")
            return None

    def list_checkpoints(self, session_id: str) -> list[dict]:
        """
        Returns checkpoint metadata for a session, newest first.

        FIXED: previously a DB read failure here was indistinguishable from
        "this session genuinely has zero checkpoints" — both returned `[]`
        with zero logging. A caller (e.g. the /api/checkpoints/{session_id}
        endpoint) would show an empty list to the user with no way to tell
        whether checkpointing is broken or just hasn't run yet. We still
        return [] on failure (changing the return type here would break the
        API contract), but now we log it at error level so it's actually
        visible in production instead of invisible by design.
        """
        try:
            conn = self._db_connect()
            try:
                rows = conn.execute(
                    """SELECT checkpoint_id, created_at, context_pct, trigger, message_count
                       FROM checkpoints WHERE session_id=? ORDER BY created_at DESC""",
                    (session_id,),
                ).fetchall()
            finally:
                conn.close()
            return [
                {"checkpoint_id": r[0], "created_at": r[1], "context_pct": r[2],
                 "trigger": r[3], "message_count": r[4]}
                for r in rows
            ]
        except Exception as e:
            logger.error(f"Checkpoint list query failed for session {session_id}: {e}")
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
