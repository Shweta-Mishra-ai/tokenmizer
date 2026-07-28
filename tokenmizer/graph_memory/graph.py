"""
Graph Memory — the core of TokenMizer's context continuity.

Key fixes over V3:
- Node deduplication by normalized label+type
- LLM-powered extraction (haiku/gpt-4o-mini) with heuristic fallback
- Full message history extraction (not just last 10)
- Incremental extraction (skip already-processed messages)
- New node types: ENVIRONMENT, GOAL, TEST, ENDPOINT, SCHEMA
- Graph pruning / aging
- Secret redaction on every write
- SQLite persistence (survives restarts)

Module layout (split for maintainability — see types.py / helpers.py):
- types.py:   NodeType, NodeStatus, EdgeType, MemoryNode, MemoryEdge, DecisionTransition
- helpers.py: _content_to_text, _infer_trigger, _extract_evidence_from_text
- graph.py:   GraphMemory (this file) — all extraction/query/persistence logic

All names below are re-exported here for backward compatibility:
existing code doing `from tokenmizer.graph_memory.graph import NodeType` etc.
continues to work unchanged.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

from tokenmizer.graph_memory.helpers import (
    _content_to_text,
    _extract_evidence_from_text,
    _infer_trigger,
)
from tokenmizer.graph_memory.types import (
    DecisionTransition,
    EdgeType,
    MemoryEdge,
    MemoryNode,
    NodeStatus,
    NodeType,
)

__all__ = [
    "GraphMemory",
    "NodeType", "NodeStatus", "EdgeType",
    "MemoryNode", "MemoryEdge", "DecisionTransition",
    "_content_to_text", "_infer_trigger", "_extract_evidence_from_text",
]

logger = logging.getLogger(__name__)


# Lazy import to avoid circular dependency
def _get_validator():
    from tokenmizer.graph_memory.validator import get_validator
    return get_validator()


# ── Graph ────────────────────────────────────────────────────────────────────

class GraphMemory:
    """
    In-process graph with SQLite persistence.
    Survives process restarts. One DB file per storage_dir.
    """

    # Days a decision stays SUPERSEDED before aging into ARCHIVED
    # (measured from the moment of supersession — see apply_importance_decay).
    ARCHIVE_SUPERSEDED_AFTER_DAYS: float = 7.0

    def __init__(self, session_id: str, storage_dir: str = "./checkpoints"):
        self.session_id = session_id
        self._nodes: dict[str, MemoryNode] = {}
        self._edges: list[MemoryEdge] = []
        self._transitions: list[DecisionTransition] = []   # full causal history
        self._processed_hashes: set[str] = set()
        self._schema_version = 1  # increment when storage format changes
        # Counts non-fatal decision-contradiction-check failures (see add_node).
        # Persistently non-zero means the supersede-tracking feature is
        # broken, even though node creation itself keeps working.
        self._decision_tracking_failures = 0
        # True if the SQLite DB could not be reinitialized after corruption —
        # the graph is running in-memory-only with no durable persistence.
        self._persistence_broken = False
        # Dirty-tracking for _persist() — see that method's docstring.
        # Starts True so the first persist() call after construction always
        # writes; cleared only after a confirmed successful write.
        self._dirty = True
        self._db_path = Path(storage_dir) / "graph_memory.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._safe_init_db()
        self._load()
        self._load_transitions()

    def _safe_init_db(self) -> None:
        """Initialize DB, deleting corrupt file if necessary."""
        try:
            self._init_db()
        except Exception:
            logger.warning(f"DB corrupt or unreadable — recreating: {self._db_path}")
            try:
                self._db_path.unlink(missing_ok=True)
            except Exception as del_err:
                logger.error(f"Could not delete corrupt graph DB: {del_err}")
            try:
                self._init_db()
            except Exception as e:
                logger.error(
                    f"Cannot initialize DB after cleanup for {self.session_id}: {e} "
                    "— running with in-memory graph only (data won't persist)"
                )
                # FIXED: previously this was a dead end — logged once at
                # startup and then silently true for the rest of the
                # process's life with no way to query it. See _load()'s
                # matching reinit-failure path for the same fix.
                self._persistence_broken = True

    # ── DB ──────────────────────────────────────────────────────────────────

    def _db_connect(self) -> "sqlite3.Connection":
        """
        Open a SQLite connection with safe concurrent settings:
        - WAL journal mode: readers don't block writers, writers don't block readers
        - 5s timeout: prevents instant failure when another process holds a write lock
        - check_same_thread=False: safe because we serialize via asyncio session locks
        """
        conn = sqlite3.connect(str(self._db_path), timeout=5.0, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")  # WAL + NORMAL = safe + fast
        except Exception:
            # Close before re-raising: a failed PRAGMA (corrupt DB file)
            # would otherwise leak the handle, and an open handle blocks
            # the delete-and-recreate recovery path on Windows.
            conn.close()
            raise
        return conn

    def _init_db(self) -> None:
        conn = self._db_connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graphs (
                    session_id TEXT PRIMARY KEY,
                    nodes_json TEXT NOT NULL,
                    edges_json TEXT NOT NULL,
                    processed_hashes TEXT NOT NULL DEFAULT '[]',
                    updated_at REAL NOT NULL
                )
            """)
            # Separate table for decision transitions — full causal story
            # Kept separate so it survives graph pruning and is queryable independently
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_transitions (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    from_decision_id TEXT NOT NULL,
                    to_decision_id TEXT NOT NULL,
                    from_label TEXT NOT NULL,
                    to_label TEXT NOT NULL,
                    trigger TEXT NOT NULL DEFAULT '',
                    reason TEXT NOT NULL DEFAULT '',
                    evidence TEXT NOT NULL DEFAULT '',
                    confidence_delta REAL NOT NULL DEFAULT 0.0,
                    timestamp REAL NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()


    def _persist_transition(self, t: DecisionTransition) -> None:
        """Persist a single DecisionTransition to its own SQLite table."""
        try:
            conn = self._db_connect()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO decision_transitions
                       (id, session_id, from_decision_id, to_decision_id,
                        from_label, to_label, trigger, reason, evidence,
                        confidence_delta, timestamp)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (t.id, t.session_id, t.from_decision_id, t.to_decision_id,
                     t.from_label, t.to_label, t.trigger, t.reason,
                     t.evidence, t.confidence_delta, t.timestamp),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Transition persist failed: {e}")

    def get_transitions(self) -> list[DecisionTransition]:
        """Return all decision transitions for this session, newest first."""
        return sorted(self._transitions, key=lambda t: t.timestamp, reverse=True)

    def _load_transitions(self) -> None:
        """Load transitions from SQLite into memory."""
        try:
            conn = self._db_connect()
            try:
                rows = conn.execute(
                    "SELECT id,session_id,from_decision_id,to_decision_id,"
                    "from_label,to_label,trigger,reason,evidence,"
                    "confidence_delta,timestamp "
                    "FROM decision_transitions WHERE session_id=?",
                    (self.session_id,),
                ).fetchall()
                self._transitions = [
                    DecisionTransition(
                        id=r[0], session_id=r[1],
                        from_decision_id=r[2], to_decision_id=r[3],
                        from_label=r[4], to_label=r[5],
                        trigger=r[6], reason=r[7], evidence=r[8],
                        confidence_delta=r[9], timestamp=r[10],
                    )
                    for r in rows
                ]
            finally:
                conn.close()
        except Exception as e:
            # Distinguish the benign first-run case (table not created yet,
            # debug) from real failures such as corruption or schema
            # mismatch, which must be visible at default log levels.
            if "no such table" in str(e).lower():
                logger.debug(f"Transition load skipped (first run, table not "
                             f"created yet): {e}")
            else:
                logger.warning(f"Transition load FAILED for session "
                               f"{self.session_id} — decision supersession "
                               f"history unavailable this session: "
                               f"{type(e).__name__}: {e}")
            self._transitions = []

    def _persist(self, force: bool = False) -> bool:
        """
        Persist the full graph (all nodes + edges) as JSON to SQLite.

        Returns True if the graph's current state is confirmed safely on
        disk — either the write just succeeded, or nothing was dirty so
        the last successful write already covers it — and False if a
        write was attempted and failed.

        FIXED (TM-12): this used to return None unconditionally, catching
        and logging its own exceptions internally without ever re-raising.
        api/app.py's graph-cache eviction path (_graph_cache_touch) wraps
        `evicted_graph._persist()` in a try/except with a retry-once loop
        specifically to handle transient failures — but since this method
        never raised, that except clause could never fire, `persisted =
        True` was set unconditionally on the very first call, and the
        retry (plus the record_silent_failure metric it guards) was dead
        code. A detailed comment there described that retry/alerting
        behavior as implemented; it never executed once. Returning bool
        lets the caller check the actual outcome instead of relying on an
        exception that was never going to arrive.

        KNOWN SCALING LIMITATION (documented, not silently shipped as if
        it were fine): this rewrites EVERY node and edge as JSON on every
        call, even when only 1-2 nodes actually changed. Cost is O(total
        node count) per persist, and persist is called once per chat turn
        in extract_from_messages(). The existing 200-node auto-prune cap
        (see prune()) is itself evidence this was already a known
        bottleneck — it caps the damage rather than fixing the cause.

        Why this isn't rewritten to a proper per-node table in this pass:
        that's a real schema migration (one row per node/edge instead of
        one JSON blob per session), and shipping a migration without being
        able to run it against real persisted data in this environment
        (no app runtime available here — see repo's TESTING.md) is exactly
        the kind of "looks fixed, silently corrupts production data" risk
        this audit is supposed to eliminate, not introduce. Tracked as a
        documented follow-up: migrate `graphs.nodes_json` blob storage to
        a `graph_nodes(session_id, node_id, data_json, updated_at)` table
        with per-node INSERT OR REPLACE, validated against a copy of real
        checkpoint data before rollout.

        What IS fixed here: a dirty flag so we skip the rewrite entirely
        when nothing changed since the last successful persist (e.g. a
        message produced zero new/updated nodes — common for short
        acknowledgement turns). This doesn't fix the O(n) cost when a
        write IS needed, but it does eliminate redundant full-rewrites,
        which in practice is a meaningful fraction of calls.

        force=True bypasses the dirty check. Required for callers that
        mutate node/edge state directly without going through add_node()
        (which sets the dirty flag) — e.g. the /api/decision/invalidate
        endpoint flips `node.status` directly, then must force a write or
        the change would silently never be saved.
        """
        if not self._dirty and not force:
            return True
        try:
            conn = self._db_connect()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO graphs
                       (session_id, nodes_json, edges_json, processed_hashes, updated_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        self.session_id,
                        json.dumps([asdict(n) for n in self._nodes.values()]),
                        json.dumps([asdict(e) for e in self._edges]),
                        json.dumps(list(self._processed_hashes)),
                        time.time(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
            self._dirty = False  # only clear on confirmed success
            return True
        except Exception as e:
            logger.error(f"Graph persist failed for {self.session_id}: {e}")
            # _dirty stays True — next call (even non-forced) will retry
            # the full write rather than silently giving up on it forever.
            return False

    def _load(self) -> None:
        try:
            conn = self._db_connect()
            try:
                row = conn.execute(
                    "SELECT nodes_json, edges_json, processed_hashes FROM graphs WHERE session_id=?",
                    (self.session_id,),
                ).fetchone()
            finally:
                conn.close()
            if not row:
                return
            nodes_data = json.loads(row[0])
            edges_data = json.loads(row[1])

            # FIXED — real bug, found while writing a proper (non-vacuous) test
            # for tests/chaos/test_recovery.py::test_partial_write_recovery.
            # processed_hashes used to be parsed inline with nodes/edges, all
            # inside the same try block. If processed_hashes was corrupted
            # (e.g. a partial/interrupted write), json.loads() on it raised
            # BEFORE the node-population loop below ever ran — so a session
            # with perfectly valid nodes_json still lost every node on reload,
            # just because the unrelated hashes field was bad. That directly
            # contradicts this method's whole purpose (recover what's good).
            # Isolating this parse means a corrupt hash set only costs you
            # incremental-extraction dedup (some messages get re-processed —
            # harmless, add_node() already dedupes), not your entire graph.
            try:
                self._processed_hashes = set(json.loads(row[2]))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    f"processed_hashes corrupted for {self.session_id}, "
                    f"resetting (nodes/edges are unaffected): {e}"
                )
                self._processed_hashes = set()

            # CRITICAL: restore enum types on load. asdict() serializes
            # NodeType/NodeStatus (str-Enums) to plain strings; without
            # converting back, every reloaded node has type/status as `str`.
            # Because they're str-Enums, equality checks still pass — which
            # HID this bug — but any `.value` access crashes ('str' object
            # has no attribute 'value'). Concretely: after a server restart,
            # every checkpoint of a reloaded session returned HTTP 500.
            # Found by the MCP e2e check, not by unit tests, because unit
            # tests reloaded graphs but never then called `.type.value`.
            for nd in nodes_data:
                nd.pop("_evicted", None)
                try:
                    nd["type"] = NodeType(nd["type"])
                    nd["status"] = NodeStatus(nd["status"])
                except (ValueError, KeyError) as conv_err:
                    logger.warning(
                        f"Skipping node with unknown type/status during load "
                        f"for {self.session_id}: {conv_err} — {nd.get('id')}"
                    )
                    continue
                n = MemoryNode(**{k: v for k, v in nd.items() if k != "_evicted"})
                self._nodes[n.id] = n
            for ed in edges_data:
                try:
                    ed["type"] = EdgeType(ed["type"])
                except (ValueError, KeyError) as conv_err:
                    logger.warning(
                        f"Skipping edge with unknown type during load "
                        f"for {self.session_id}: {conv_err}"
                    )
                    continue
                self._edges.append(MemoryEdge(**ed))
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
            logger.warning(f"Corrupted DB for {self.session_id} — starting fresh: {e}")
            self._nodes = {}
            self._edges = []
            self._processed_hashes = set()
            # Re-initialize the DB file
            try:
                self._db_path.unlink(missing_ok=True)
                self._init_db()
            except Exception as reinit_err:
                logger.error(
                    f"Graph DB reinit failed for {self.session_id}: {reinit_err} "
                    "— running with in-memory graph only (data won't persist)"
                )
                # FIXED: this is the worst-case path — persistence is
                # completely broken for this session going forward, but
                # previously the only trace of that fact was a log line.
                # Surfacing it via stats() means a health-check script (or
                # a human looking at /api/graph/{session_id}) can detect
                # "this session has no durable memory" instead of finding
                # out only after a restart wipes everything.
                self._persistence_broken = True
        except Exception as e:
            logger.warning(f"Graph load failed for {self.session_id}: {e}")

    # ── Nodes ────────────────────────────────────────────────────────────────

    def _node_id(self, node_type: str, label: str) -> str:
        normalized = f"{node_type}:{label.lower().strip()}"
        return hashlib.sha1(normalized.encode()).hexdigest()[:12]

    def _normalize_label(self, label: str) -> str:
        return label.lower().strip().rstrip(".,!?")

    def add_node(
        self,
        node_type: NodeType,
        label: str,
        status: NodeStatus = NodeStatus.PENDING,
        summary: str = "",
        importance: float = 0.5,
        confidence: float = 0.7,
    ) -> str:
        from tokenmizer.security.redaction import redact_node
        label, summary = redact_node(label, summary)

        stored_label = label[:120]
        stored_summary = summary[:300]
        norm = self._normalize_label(stored_label)
        node_id = self._node_id(node_type.value, norm)

        if node_id in self._nodes:
            # Dedup: update existing node instead of creating duplicate
            existing = self._nodes[node_id]
            existing.touch()
            self._dirty = True  # touch() always changes updated_at, must persist
            # Only upgrade status (completed > in_progress > pending)
            status_rank = {
                NodeStatus.PENDING: 0,
                NodeStatus.IN_PROGRESS: 1,
                NodeStatus.COMPLETED: 2,
                NodeStatus.FAILED: 3,
                NodeStatus.ARCHIVED: 4,
                NodeStatus.SUPERSEDED: 5,
                NodeStatus.MODIFIED: 5,    # alias for SUPERSEDED
                NodeStatus.INVALIDATED: 6,
            }
            if status_rank.get(status, 0) > status_rank.get(existing.status, 0):
                existing.status = status
            if stored_summary and not existing.summary:
                existing.summary = stored_summary
            return node_id

        # Fuzzy same-decision merge. Exact-hash dedup above only catches
        # identical normalized labels; the extractor can emit near-duplicate
        # variants of one decision from a single message. Left as separate
        # nodes they share a topic and one would supersede the other — a
        # spurious self-supersession. Merge into the existing node instead:
        # prefer the longer (more specific) label, backfill the summary.
        if node_type == NodeType.DECISION:
            from tokenmizer.graph_memory.decision_tracker import _is_same_decision
            for ex_id, ex in self._nodes.items():
                if ex.type != NodeType.DECISION or ex._evicted:
                    continue
                if _is_same_decision(label, ex.label):
                    ex.touch()
                    self._dirty = True
                    # Same status-upgrade-only rule as the exact-dedup branch
                    # (a COMPLETED re-add must not resurrect a SUPERSEDED node)
                    _rank = {
                        NodeStatus.PENDING: 0, NodeStatus.IN_PROGRESS: 1,
                        NodeStatus.COMPLETED: 2, NodeStatus.FAILED: 3,
                        NodeStatus.ARCHIVED: 4, NodeStatus.SUPERSEDED: 5,
                        NodeStatus.MODIFIED: 5, NodeStatus.INVALIDATED: 6,
                    }
                    if _rank.get(status, 0) > _rank.get(ex.status, 0):
                        ex.status = status
                    if len(stored_label) > len(ex.label) and status == ex.status:
                        ex.label = stored_label
                    if stored_summary and not ex.summary:
                        ex.summary = stored_summary
                    return ex_id

        # Validate before inserting — reject noise and low-confidence nodes.
        # confidence != 0.7 (the parameter default) means the caller
        # supplied an explicit value; for extraction-sourced decisions that
        # is merge()'s corroboration tier, which the validator blends into
        # its own score (see validator.validate).
        validator = _get_validator()
        result = validator.validate(
            label=label,
            node_type=node_type.value,
            summary=summary,
            extractor_confidence=confidence if confidence != 0.7 else None,
        )
        if not result.accepted:
            logger.debug(f"Node rejected: {label!r} ({result.rejection_reason})")
            return ""  # empty string = rejected, callers must check

        # Apply type correction if validator detected mismatch
        if result.corrected_type:
            try:
                node_type = NodeType(result.corrected_type)
                node_id = self._node_id(node_type.value, norm)
            except ValueError:
                pass  # keep original type if correction is unknown

        node = MemoryNode(
            id=node_id,
            type=node_type,
            label=stored_label,
            status=status,
            summary=stored_summary,
            importance=importance,
            confidence=confidence if confidence != 0.7 else result.confidence,
        )
        self._nodes[node_id] = node
        self._dirty = True

        # Decision contradiction detection — capture full transition story
        if node_type == NodeType.DECISION and status == NodeStatus.COMPLETED:
            try:
                from tokenmizer.graph_memory.decision_tracker import (
                    find_contested_decisions,
                    find_contradicting_decisions,
                )
                to_supersede = find_contradicting_decisions(
                    label, summary, self._nodes
                )
                for old_id in to_supersede:
                    if old_id != node_id and old_id in self._nodes:
                        old_node = self._nodes[old_id]
                        old_confidence = old_node.confidence

                        # Mark old decision superseded
                        old_node.status = NodeStatus.SUPERSEDED
                        old_node.valid_until = time.time()

                        # Build full transition object
                        # Evidence: prefer explicit "|" separator, else extract from summary
                        parts = (summary or "").split("|", 1)
                        reason_text = parts[0].strip()
                        evidence_text = parts[1].strip() if len(parts) > 1 else ""

                        # Auto-extract evidence from summary if not explicit
                        if not evidence_text and summary:
                            evidence_text = _extract_evidence_from_text(summary)

                        trigger = _infer_trigger(old_node.label, label, summary)

                        transition = DecisionTransition(
                            id=f"tr_{old_id[:8]}_{node_id[:8]}",
                            session_id=self.session_id,
                            from_decision_id=old_id,
                            to_decision_id=node_id,
                            from_label=old_node.label,
                            to_label=label,
                            trigger=trigger,
                            reason=reason_text,
                            evidence=evidence_text,
                            confidence_delta=round(confidence - old_confidence, 3),
                        )
                        self._transitions.append(transition)
                        self._persist_transition(transition)

                        old_node.summary = (
                            f"Superseded by: {label[:60]}"
                            + (f" — {reason_text[:40]}" if reason_text else "")
                        )
                        self.add_edge(node_id, old_id, EdgeType.SUPERSEDES, weight=1.0)
                        logger.info(
                            f"Decision transition: {old_node.label!r} → {label!r}"
                            f" | trigger: {trigger[:40]}"
                        )

                # CONTESTED (TM-09): decisions sharing a topic bucket with
                # the new one, but NOT confident enough to supersede (see
                # _same_slot in decision_tracker.py) — e.g. "Use
                # PostgreSQL for primary user data" and "Use SQLite for
                # the local offline cache" both classify as "database"
                # but plausibly serve different purposes. Rather than
                # silently guessing (which risks destroying correct
                # information) or silently doing nothing (which hides the
                # ambiguity), both sides are flagged CONTESTED and linked
                # so a human or the LLM can resolve it explicitly. No
                # DecisionTransition is recorded — this is an unresolved
                # conflict, not a causal replacement.
                contested = find_contested_decisions(
                    label, summary, self._nodes, exclude_ids=frozenset(to_supersede),
                )
                for other_id in contested:
                    if other_id == node_id or other_id not in self._nodes:
                        continue
                    other_node = self._nodes[other_id]
                    other_node.status = NodeStatus.CONTESTED
                    node.status = NodeStatus.CONTESTED
                    self.add_edge(node_id, other_id, EdgeType.CONFLICTS_WITH, weight=1.0)
                    self.add_edge(other_id, node_id, EdgeType.CONFLICTS_WITH, weight=1.0)
                    logger.info(
                        f"Decision contested: {other_node.label!r} vs {label!r} "
                        f"— same topic, different purpose; flagging both instead "
                        f"of silently superseding one"
                    )
            except Exception as e:
                # Intentionally non-fatal: a bug in contradiction detection
                # must not block creating the new decision node itself —
                # the node is more important than the supersede-tracking
                # metadata around it.
                #
                # FIXED: previously logged at `debug` (off by default in
                # production), meaning this core advertised feature —
                # "decision transition tracking" / "Changed X → Y" in
                # resume context — could silently stop working entirely
                # and nobody would notice until they wondered why resume
                # context never showed any decision changes. Bumped to
                # `warning` and counted on the instance (surfaced via
                # stats(), see below) so persistent failures are visible
                # to anyone inspecting graph health, not just to someone
                # who happens to be tailing logs at warning level.
                logger.warning(
                    f"Decision contradiction check failed for node {node_id} "
                    f"(non-fatal — node was still created): {e}"
                )
                self._decision_tracking_failures += 1

        return node_id

    def add_edge(
        self, source_id: str, target_id: str, edge_type: EdgeType, weight: float = 1.0
    ) -> None:
        # No duplicate edges
        for e in self._edges:
            if e.source_id == source_id and e.target_id == target_id and e.type == edge_type:
                return
        self._edges.append(MemoryEdge(source_id=source_id, target_id=target_id,
                                       type=edge_type, weight=weight))
        self._dirty = True

    # ── Extraction ───────────────────────────────────────────────────────────

    def _msg_hash(self, msg: dict) -> str:
        """
        Hash a message for dedup tracking.
        Handles non-string content: None (empty), list (multimodal — extract text
        parts), dict, or any other type (str() fallback).
        """
        content = msg.get("content", "")
        text = _content_to_text(content)
        return hashlib.sha1(text[:500].encode()).hexdigest()[:16]

    def extract_from_messages(
        self,
        messages: list[dict],
        incremental: bool = True,
        extracted_data: dict | None = None,
    ) -> None:
        """
        Update graph from messages.

        Pipeline:
          1. If extracted_data is provided (from LLM/HybridExtractor) — use it directly.
          2. Otherwise run _heuristic_extract() as fallback.
        """
        if incremental:
            new_messages = [m for m in messages
                           if self._msg_hash(m) not in self._processed_hashes]
            if not new_messages:
                return
        else:
            new_messages = messages

        # Auto-select sliding window for long sessions
        # For sessions > 30 messages: only extract WIP/errors from last 20
        window_size = 20 if len(messages) > 30 else 0

        # Use provided data (from LLM pipeline) or run HybridExtractor heuristic pass
        if extracted_data is not None:
            data = extracted_data
        else:
            from tokenmizer.graph_memory.hybrid_extractor import get_hybrid_extractor
            _he = get_hybrid_extractor()
            _extracted = _he.heuristic_extract(new_messages, window_size=window_size)
            data = {
                "goals":        _extracted.goals,
                "tasks":        (
                    [{"label": t, "status": "completed"}   for t in _extracted.tasks_done] +
                    [{"label": t, "status": "in_progress"} for t in _extracted.tasks_wip]  +
                    [{"label": t, "status": "pending"}     for t in _extracted.tasks_todo]
                ),
                "decisions":    _extracted.decisions,
                "files":        _extracted.files,
                "errors":       _extracted.errors,
                "dependencies": _extracted.dependencies,
                "environments": _extracted.environments,
                "endpoints":    _extracted.endpoints,
                "schemas":      _extracted.schemas,
                "superseded":   _extracted.superseded,
            }
        self._apply_extracted(data, new_messages)

        for m in new_messages:
            self._processed_hashes.add(self._msg_hash(m))
        if new_messages:
            self._dirty = True  # processed_hashes changed even if no nodes did

        # Cap processed_hashes — for very long sessions (1000+ turns), this set
        # would otherwise grow unbounded (each hash ~16 bytes, but still).
        # When over cap, rebuild from the most recent messages only.
        # Effect: very old messages may be re-scanned on restart, but since
        # their content is already in the graph, add_node() dedup makes
        # re-extraction a safe no-op.
        _MAX_PROCESSED_HASHES = 500
        if len(self._processed_hashes) > _MAX_PROCESSED_HASHES:
            self._processed_hashes = {
                self._msg_hash(m) for m in messages[-_MAX_PROCESSED_HASHES:]
            }
            self._dirty = True  # processed_hashes is part of the persisted row

        # Apply importance decay — completed tasks fade, superseded decisions fade
        # Active decisions and goals never decay
        decayed = self.apply_importance_decay()
        if decayed:
            logger.debug(f"Importance decay applied to {len(decayed)} nodes")

        # Auto-prune: if graph has grown large, remove low-importance old nodes.
        # Runs only when over threshold — cheap no-op for typical sessions.
        if len(self._nodes) > 200:
            pruned = self.prune(max_nodes=200)
            if pruned:
                logger.debug(f"Auto-pruned {pruned} nodes (graph exceeded 200 nodes)")

        self._persist()

    def _apply_extracted(self, data: dict, messages: list[dict]) -> None:
        """
        Apply extracted structured data to the graph.

        Edge rule: edges are created only between semantically related nodes,
        NOT by accident-of-order (previous version used task_ids[-3:] which
        linked any task to any file extracted in the same message — wrong).

        Relationship logic:
          - decision → task: only if decision label shares ≥1 meaningful word with task
          - task → file: only if file name appears in task label or vice versa
          - file → endpoint: only if endpoint label shares a path segment with file name
        """
        # Collect accepted node IDs by type for relationship inference
        goal_ids: list[str] = []
        task_ids: list[str] = []
        file_ids: list[str] = []
        decision_ids: list[str] = []

        # Goals
        for goal in data.get("goals", []):
            if goal:
                nid = self.add_node(NodeType.GOAL, goal, NodeStatus.IN_PROGRESS, importance=1.0)
                if nid:
                    goal_ids.append(nid)

        # Tasks
        status_map = {
            "completed": NodeStatus.COMPLETED,
            "in_progress": NodeStatus.IN_PROGRESS,
            "failed": NodeStatus.FAILED,
        }
        for t in data.get("tasks", []):
            label = t.get("label", "")
            if not label or len(label) < 5:
                continue
            status = status_map.get(t.get("status", "pending"), NodeStatus.PENDING)
            importance = 0.8 if status == NodeStatus.COMPLETED else 0.6
            nid = self.add_node(NodeType.TASK, label, status, importance=importance)
            if nid:
                task_ids.append(nid)
                # Tasks are part of the session goal
                for gid in goal_ids:
                    self.add_edge(nid, gid, EdgeType.PART_OF)

        # Decisions — linked to tasks that share vocabulary
        for d in data.get("decisions", []):
            label = d.get("label", "")
            if not label or len(label) < 5:
                continue
            summary = d.get("rationale", d.get("reason", ""))
            # Use per-item confidence from merge() if provided (corroboration signal).
            # Fallback: 0.9 for explicit decisions (high-value nodes).
            node_confidence = float(d.get("confidence", 0.9))
            nid = self.add_node(NodeType.DECISION, label, NodeStatus.COMPLETED,
                                summary=summary, importance=0.9,
                                confidence=node_confidence)
            if nid:
                decision_ids.append(nid)
                # Link to tasks if they share meaningful vocabulary (with alias expansion)
                decision_words = self._expand_with_aliases(
                    self._meaningful_words(label)
                )
                for tid in task_ids:
                    task_node = self._nodes.get(tid)
                    if task_node:
                        task_words = self._expand_with_aliases(
                            self._meaningful_words(task_node.label)
                        )
                        if decision_words & task_words:
                            self.add_edge(nid, tid, EdgeType.RELATED_TO)

                # SUPERSEDES edge: link new decision to any SUPERSEDED decision
                # that shares topic words — enables "changed from X to Y" in resume
                for existing_id, existing_node in list(self._nodes.items()):
                    if (existing_id != nid
                            and existing_node.type == NodeType.DECISION
                            and existing_node.status == NodeStatus.SUPERSEDED):
                        existing_words = self._expand_with_aliases(
                            self._meaningful_words(existing_node.label)
                        )
                        if decision_words & existing_words:
                            self.add_edge(nid, existing_id, EdgeType.SUPERSEDES)

        # Files — linked to tasks only if file name appears in task description
        for f in data.get("files", []):
            if not f or len(f) < 3:
                continue
            nid = self.add_node(NodeType.FILE, f, NodeStatus.IN_PROGRESS, importance=0.7)
            if nid:
                file_ids.append(nid)
                file_stem = f.split("/")[-1].split(".")[0].lower()
                for tid in task_ids:
                    task_node = self._nodes.get(tid)
                    if task_node and file_stem and file_stem in task_node.label.lower():
                        self.add_edge(tid, nid, EdgeType.IMPLEMENTS)

        # Errors — handle both str and dict formats
        for e in data.get("errors", []):
            if isinstance(e, str):
                label, resolved = e, False
            else:
                label, resolved = e.get("label", ""), e.get("resolved", False)
            if not label:
                continue
            status = NodeStatus.COMPLETED if resolved else NodeStatus.FAILED
            importance = 0.5 if resolved else 0.9
            err_nid = self.add_node(NodeType.ERROR, label, status, importance=importance)
            if err_nid:
                for fid in file_ids:
                    file_node = self._nodes.get(fid)
                    if file_node and file_node.label.split("/")[-1] in label:
                        self.add_edge(err_nid, fid, EdgeType.RELATED_TO)

        # Dependencies (no edges — standalone nodes)
        for dep in data.get("dependencies", []):
            if dep and len(dep) > 1:
                self.add_node(NodeType.DEPENDENCY, dep, NodeStatus.COMPLETED, importance=0.6)

        # Environment (no edges — standalone nodes)
        for env in data.get("environments", data.get("environment", [])):
            if env:
                self.add_node(NodeType.ENVIRONMENT, env, NodeStatus.COMPLETED, importance=0.8)

        # Endpoints — linked to files only when they share a path segment
        for ep in data.get("endpoints", []):
            if not ep:
                continue
            ep_nid = self.add_node(NodeType.ENDPOINT, ep, NodeStatus.COMPLETED, importance=0.7)
            if ep_nid:
                ep_parts = set(ep.lower().replace("/", " ").split())
                for fid in file_ids:
                    file_node = self._nodes.get(fid)
                    if file_node:
                        file_parts = self._meaningful_words(file_node.label)
                        if ep_parts & file_parts:
                            self.add_edge(fid, ep_nid, EdgeType.IMPLEMENTS)

        # Schemas
        for schema in data.get("schemas", []):
            if schema:
                self.add_node(NodeType.SCHEMA, schema, NodeStatus.COMPLETED, importance=0.7)

    _STOP_WORDS = frozenset({
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "is", "are", "was", "were", "be", "been",
        "have", "has", "do", "does", "will", "would", "could", "should",
        "this", "that", "it", "we", "i", "you", "they",
        # NOTE: "use" and "using" intentionally NOT in stop words —
        # they appear in decision labels like "Use PostgreSQL for sessions"
        # and removing them kills edge matching between decisions and tasks.
    })

    # Tech aliases: maps common abbreviations/variants to canonical tokens
    # Allows "Use PG" to match task "Set up PostgreSQL database"
    _TECH_ALIASES: dict[str, frozenset] = {
        "postgres":     frozenset({"postgres", "postgresql", "pg", "psql"}),
        "postgresql":   frozenset({"postgres", "postgresql", "pg", "psql"}),
        "pg":           frozenset({"postgres", "postgresql", "pg", "psql"}),
        "mongo":        frozenset({"mongo", "mongodb"}),
        "mongodb":      frozenset({"mongo", "mongodb"}),
        "redis":        frozenset({"redis", "cache", "caching"}),
        "jwt":          frozenset({"jwt", "token", "auth", "authentication"}),
        "auth":         frozenset({"auth", "authentication", "authorize", "jwt"}),
        "authentication": frozenset({"auth", "authentication", "authorize", "jwt"}),
        "db":           frozenset({"db", "database", "storage"}),
        "database":     frozenset({"db", "database", "storage"}),
        "api":          frozenset({"api", "endpoint", "route", "rest"}),
        "endpoint":     frozenset({"api", "endpoint", "route", "rest"}),
        "fastapi":      frozenset({"fastapi", "api", "endpoint", "route"}),
        "docker":       frozenset({"docker", "container", "containerize"}),
        "k8s":          frozenset({"k8s", "kubernetes", "cluster"}),
        "kubernetes":   frozenset({"k8s", "kubernetes", "cluster"}),
        "ts":           frozenset({"ts", "typescript"}),
        "typescript":   frozenset({"ts", "typescript"}),
        "js":           frozenset({"js", "javascript", "node", "nodejs"}),
    }

    def _expand_with_aliases(self, words: frozenset) -> frozenset:
        """Expand a word set with known tech aliases for fuzzy matching."""
        expanded = set(words)
        for w in words:
            if w in self._TECH_ALIASES:
                expanded |= self._TECH_ALIASES[w]
        return frozenset(expanded)

    def _meaningful_words(self, text: str) -> frozenset:
        """Extract meaningful words from text for semantic edge linking."""
        words = set(text.lower().split())
        # Remove stop words, punctuation, and very short words
        return frozenset(
            w.strip(".,!?:;()[]") for w in words
            if len(w) > 3 and w not in self._STOP_WORDS
        )

    # ── Query ────────────────────────────────────────────────────────────────

    def query(self, task: str, top_k: int = 12) -> list[MemoryNode]:
        """
        Keyword + importance + type-boosted ranked retrieval.
        Uses alias expansion so 'auth' matches 'authentication', 'PG' matches 'PostgreSQL'.
        Type boost: DECISION/GOAL nodes score 20% higher when relevant.
        """
        query_words = self._expand_with_aliases(
            frozenset(w.strip(".,!?:;()[]").lower() for w in task.split() if len(w) > 2)
        )

        # Type boost factors — decisions and goals are most valuable to surface
        _TYPE_BOOST = {
            NodeType.GOAL:       1.25,
            NodeType.DECISION:   1.20,
            NodeType.TASK:       1.00,
            NodeType.ERROR:      0.95,
            NodeType.FILE:       0.90,
            NodeType.ENDPOINT:   0.90,
            NodeType.SCHEMA:     0.85,
            NodeType.DEPENDENCY: 0.70,
            NodeType.ENVIRONMENT: 0.70,
        }

        scored: list[tuple[float, MemoryNode]] = []

        for node in self._nodes.values():
            if node._evicted:
                continue
            # Skip archived/superseded/invalidated — historical noise
            if node.status in (
                NodeStatus.ARCHIVED, NodeStatus.SUPERSEDED,
                NodeStatus.MODIFIED, NodeStatus.INVALIDATED,
            ):
                continue

            node_words = self._expand_with_aliases(
                frozenset(w.strip(".,!?:;()[]").lower() for w in node.label.split() if len(w) > 2)
            )
            if not node_words:
                continue

            overlap = len(query_words & node_words) / max(1, len(query_words))
            recency = 1.0 / (1.0 + node.age_days() * 0.1)
            type_boost = _TYPE_BOOST.get(node.type, 1.0)

            # Score: overlap is primary signal; importance and recency are tiebreakers
            score = (overlap * 0.6 + node.importance * 0.3 + recency * 0.1) * type_boost

            if score > 0.05:  # minimum threshold — don't return completely unrelated nodes
                scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:top_k]]

    def query_at_time(self, task: str, at_time: float, top_k: int = 12) -> list[MemoryNode]:
        """
        Return nodes that were ACTIVE at a specific point in time.

        Enables: "What did we decide last Tuesday?"

        Bug fixed: was calling query() which excludes SUPERSEDED nodes.
        A superseded decision WAS active before it was superseded.
        We must scan ALL nodes and filter by valid_from/valid_until.

        valid_from:  when the node was created (always set)
        valid_until: when it was superseded/invalidated (0.0 = still active)

        A node was active at at_time if:
          valid_from <= at_time AND (valid_until == 0 OR valid_until > at_time)
        """
        query_words = self._expand_with_aliases(
            frozenset(
                w.strip(".,!?:;()[]").lower()
                for w in (task or "").split()
                if len(w) > 2
            )
        ) if task else frozenset()

        _TYPE_BOOST = {
            NodeType.GOAL:     1.25,
            NodeType.DECISION: 1.20,
            NodeType.TASK:     1.00,
            NodeType.ERROR:    0.90,
            NodeType.FILE:     0.85,
        }

        scored: list[tuple[float, MemoryNode]] = []
        for node in self._nodes.values():
            if node._evicted:
                continue

            # Was this node active at at_time?
            was_created = node.valid_from <= at_time
            not_yet_closed = (node.valid_until == 0.0 or node.valid_until > at_time)
            if not (was_created and not_yet_closed):
                continue

            if not query_words:
                # No query — return all active nodes at that time
                scored.append((node.importance, node))
                continue

            node_words = self._expand_with_aliases(
                frozenset(
                    w.strip(".,!?:;()[]").lower()
                    for w in node.label.split()
                    if len(w) > 2
                )
            )
            if not node_words:
                continue

            overlap = len(query_words & node_words) / max(1, len(query_words))
            type_boost = _TYPE_BOOST.get(node.type, 1.0)
            score = (overlap * 0.7 + node.importance * 0.3) * type_boost

            if score > 0.05:
                scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:top_k]]

    # ── Prune ────────────────────────────────────────────────────────────────

    def apply_importance_decay(self) -> dict[str, float]:
        """Time-based importance decay. See graph_memory/pruning.py for the
        full rule set and rationale — extracted there to keep this file
        focused on core memory logic."""
        from tokenmizer.graph_memory.pruning import apply_importance_decay as _apply_decay
        return _apply_decay(self)

    def prune(
        self,
        max_nodes: int = 200,
        max_age_days: float = 60.0,
    ) -> int:
        """Remove low-importance, old, completed nodes. Preserve decisions,
        envs, goals. See graph_memory/pruning.py for the full pruning logic —
        extracted there to keep this file focused on core memory logic."""
        from tokenmizer.graph_memory.pruning import prune as _prune
        return _prune(self, max_nodes=max_nodes, max_age_days=max_age_days)

    # ── Context block ────────────────────────────────────────────────────────

    def to_context_block(self, token_budget: int = 400) -> str:
        """
        Build tiered resume context block for LLM injection.

        Priority order (truncates from bottom if over budget):
          1. Goal                    — always shown (anchor)
          2. In-progress tasks       — sorted by importance (current focus)
          3. Recent completed tasks  — top 5 by recency, not all 50
          4. Active decisions        — top 6 by importance, with rationale
          5. Recent decision changes — transition summary (not strikethrough waste)
          6. Pending tasks           — what's next
          7. Files touched           — context for file-specific questions
          8. Environment             — versions, if present
          9. Open errors             — unresolved failures

        Quality rules applied:
        - SUPERSEDED decisions: shown only as "Changed X → Y" one-liner
          (not full label — wastes tokens showing wrong answer)
        - Completed tasks: importance-weighted, capped at 5 most recent
          (full history is in SQLite, not needed in resume)
        - Similar nodes: deduplicated by normalized label prefix
        - Transitions: shown as compact lines, not repeated decision labels
        """
        sections: list[str] = []

        # ── 1. Goal ──────────────────────────────────────────────────────────
        goals = sorted(
            [n for n in self._nodes.values()
             if n.type == NodeType.GOAL and not n._evicted],
            key=lambda x: x.importance, reverse=True
        )
        if goals:
            sections.append("Goal: " + " | ".join(g.label for g in goals[:2]))

        # ── 2. In-progress tasks ──────────────────────────────────────────────
        open_tasks = sorted(
            [n for n in self._nodes.values()
             if n.type == NodeType.TASK
             and n.status == NodeStatus.IN_PROGRESS
             and not n._evicted],
            key=lambda x: x.importance, reverse=True
        )
        # ── 3. Pending tasks (next steps) ─────────────────────────────────────
        pending_tasks = sorted(
            [n for n in self._nodes.values()
             if n.type == NodeType.TASK
             and n.status == NodeStatus.PENDING
             and not n._evicted],
            key=lambda x: x.importance, reverse=True
        )
        current_work = open_tasks[:4] + pending_tasks[:2]
        if current_work:
            sections.append("Working on: " + " | ".join(t.label for t in current_work))

        # ── 4. Recent completed tasks — top 5 by recency+importance ───────────
        done = sorted(
            [n for n in self._nodes.values()
             if n.type == NodeType.TASK
             and n.status == NodeStatus.COMPLETED
             and not n._evicted],
            key=lambda x: (x.updated_at * 0.6 + x.importance * 0.4),
            reverse=True
        )
        # Deduplicate: skip if label is very similar to already-included task
        done_deduped = []
        seen_prefixes: set[str] = set()
        for t in done:
            prefix = self._normalize_label(t.label)[:20]
            if prefix not in seen_prefixes:
                done_deduped.append(t)
                seen_prefixes.add(prefix)
            if len(done_deduped) >= 6:
                break
        if done_deduped:
            sections.append("Done: " + " | ".join(t.label for t in done_deduped))

        # ── 5. Active decisions — top 6 by importance ─────────────────────────
        decisions = sorted(
            [n for n in self._nodes.values()
             if n.type == NodeType.DECISION
             and n.status == NodeStatus.COMPLETED
             and not n._evicted],
            key=lambda x: x.importance, reverse=True
        )
        if decisions:
            parts = []
            for d in decisions[:6]:
                entry = d.label
                # Include brief rationale if not redundant with label
                if d.summary and "Superseded by" not in d.summary:
                    entry += f" ({d.summary[:50]})"
                parts.append(entry)
            sections.append("Decided: " + " | ".join(parts))

        # ── 5b. Contested decisions — same topic, ambiguous whether one replaces
        # the other (see NodeStatus.CONTESTED / TM-09). Surfaced explicitly
        # rather than silently guessing which one is "current" — unlike
        # SUPERSEDED, both sides stay visible here since destroying either
        # one would risk losing correct information on weak evidence.
        contested = [
            n for n in self._nodes.values()
            if n.type == NodeType.DECISION
            and n.status == NodeStatus.CONTESTED
            and not n._evicted
        ]
        if contested:
            contested_ids = {n.id for n in contested}
            seen_pairs: set[frozenset] = set()
            lines = []
            for e in self._edges:
                if (e.type == EdgeType.CONFLICTS_WITH
                        and e.source_id in contested_ids and e.target_id in contested_ids):
                    pair = frozenset((e.source_id, e.target_id))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    a, b = self._nodes[e.source_id], self._nodes[e.target_id]
                    lines.append(f"{a.label[:45]!r} vs {b.label[:45]!r}")
            if lines:
                sections.append("Conflicting (unresolved): " + " | ".join(lines[:3]))

        # ── 6. Decision transitions — compact, no wasted tokens on wrong answer ─
        # Show as "Changed X → Y" not full old label — the old label is wrong,
        # showing it in full wastes tokens and risks LLM being confused about
        # which is current.
        recent_transitions = sorted(
            self._transitions,
            key=lambda t: t.timestamp, reverse=True
        )[:3]
        if recent_transitions:
            lines = [t.to_context_line() for t in recent_transitions]
            sections.append("Changes: " + " | ".join(lines))
        elif any(
            n.type == NodeType.DECISION
            and n.status == NodeStatus.SUPERSEDED
            and n.age_days() < 3
            and not n._evicted
            for n in self._nodes.values()
        ):
            # No transition object but recent supersede — note count only, no label
            # (showing the old wrong label wastes tokens and risks LLM confusion)
            changed_count = sum(
                1 for n in self._nodes.values()
                if n.type == NodeType.DECISION
                and n.status == NodeStatus.SUPERSEDED
                and n.age_days() < 3
                and not n._evicted
            )
            sections.append(f"Note: {changed_count} decision(s) changed recently — see graph history")

        # ── 7. Invalidated decisions — always warn ─────────────────────────────
        invalidated = [
            n for n in self._nodes.values()
            if n.type == NodeType.DECISION
            and n.status == NodeStatus.INVALIDATED
            and not n._evicted
        ]
        if invalidated:
            sections.append(
                "Avoid: " + " | ".join(f"[DO NOT USE] {n.label[:40]}" for n in invalidated[:2])
            )

        # ── 8. Files ──────────────────────────────────────────────────────────
        files = sorted(
            [n for n in self._nodes.values()
             if n.type == NodeType.FILE and not n._evicted],
            key=lambda x: x.importance, reverse=True
        )
        if files:
            sections.append("Files: " + ", ".join(f.label for f in files[:10]))

        # ── 9. Environment ────────────────────────────────────────────────────
        env_nodes = [
            n for n in self._nodes.values()
            if n.type == NodeType.ENVIRONMENT and not n._evicted
        ]
        if env_nodes:
            sections.append("Env: " + ", ".join(e.label for e in env_nodes[:4]))

        # ── 10. Open errors ───────────────────────────────────────────────────
        errors = sorted(
            [n for n in self._nodes.values()
             if n.type == NodeType.ERROR
             and n.status == NodeStatus.FAILED
             and not n._evicted],
            key=lambda x: x.importance, reverse=True
        )
        if errors:
            sections.append("Open issues: " + " | ".join(e.label for e in errors[:3]))

        block = "\n".join(sections)

        # Trim to budget — count once, char-estimate for loop, exact verify at end
        from tokenmizer.core.tokenizer import count_tokens
        total_tokens = count_tokens(block)
        if total_tokens > token_budget and sections:

            chars_per_token = len(block) / max(total_tokens, 1)
            target_chars = int(token_budget * chars_per_token * 0.92)  # 8% safety buffer
            while len("\n".join(sections)) > target_chars and sections:
                sections.pop()
            block = "\n".join(sections)
            # One final accurate tiktoken verify — trim one more section if still over
            if sections and count_tokens(block) > token_budget:
                sections.pop()
                block = "\n".join(sections)

        return block

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """
        FIXED (TM-35): this used to count _evicted nodes into node_count/
        by_type/by_status — every other consumer (query(),
        to_context_block(), both visualization.py exporters) filters
        _evicted out, so a caller comparing /api/graph/{id} (this method)
        against /api/graph/{id}/viz would see disagreeing node counts.
        """
        from tokenmizer.core.dto import GraphStatsDTO
        live_nodes = [n for n in self._nodes.values() if not n._evicted]
        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}
        confidences: list[float] = []
        for n in live_nodes:
            by_type[n.type.value] = by_type.get(n.type.value, 0) + 1
            by_status[n.status.value] = by_status.get(n.status.value, 0) + 1
            confidences.append(n.confidence)
        avg_confidence = round(sum(confidences) / max(1, len(confidences)), 3)
        dto = GraphStatsDTO(
            session_id=self.session_id,
            node_count=len(live_nodes),
            edge_count=len(self._edges),
            by_type=by_type,
            by_status=by_status,
            processed_messages=len(self._processed_hashes),
            avg_confidence=avg_confidence,
            decision_tracking_failures=self._decision_tracking_failures,
            persistence_broken=self._persistence_broken,
        )
        # Return as dict for JSON serialization — DTO used for type safety at boundary
        from dataclasses import asdict
        return asdict(dto)

    # ── Visualization exports (see visualization.py) ──────────────────────────

    def to_vis_json(self) -> dict:
        """D3-compatible JSON export. Full implementation in visualization.py."""
        from tokenmizer.graph_memory.visualization import to_vis_json
        return to_vis_json(self)

    def to_obsidian_canvas(self) -> dict:
        """Obsidian Canvas export. Full implementation in visualization.py."""
        from tokenmizer.graph_memory.visualization import to_obsidian_canvas
        return to_obsidian_canvas(self)
