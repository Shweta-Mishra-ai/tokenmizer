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

Module layout (split for maintainability):
- types.py:          NodeType, NodeStatus, EdgeType, MemoryNode, MemoryEdge, DecisionTransition
- helpers.py:         _content_to_text, _infer_trigger, _extract_evidence_from_text
- persistence.py:     SQLite init/connect, transition persistence, save/load round-trip
- pruning.py:         importance decay + node-count cap enforcement
- context_block.py:   the tiered, token-budgeted resume context builder
- visualization.py:   D3 / Obsidian Canvas exports
- graph.py (this file): GraphMemory — node/edge CRUD, extraction, query.
  Delegates persistence/pruning/context-block/visualization work to the
  modules above via one-line wrapper methods, so existing callers
  (`graph._persist()`, `graph.prune()`, `graph.to_context_block()`, etc.)
  are unaffected.

All names below are re-exported here for backward compatibility:
existing code doing `from tokenmizer.graph_memory.graph import NodeType` etc.
continues to work unchanged.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from pathlib import Path

from tokenmizer.graph_memory.helpers import (
    _content_to_text,
    _extract_evidence_from_text,
    _infer_trigger,
)
from tokenmizer.graph_memory.types import (
    INACTIVE_STATUSES,
    DecisionTransition,
    EdgeType,
    MemoryEdge,
    MemoryNode,
    NodeStatus,
    NodeType,
)

__all__ = [
    "GraphMemory",
    "NodeType", "NodeStatus", "EdgeType", "INACTIVE_STATUSES",
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
        # True if stored data was destroyed or displaced during recovery
        # (a session row dropped, or the shared DB file quarantined).
        # Distinct from _persistence_broken, which is about FUTURE writes:
        # this says "memory that existed is gone." Both are surfaced via
        # stats() so a health check can tell "empty because new" from
        # "empty because we lost it."
        self._data_loss_detected = False
        # True if the last _load() could not read the DB at all (e.g. lock
        # contention). The in-memory graph is empty but NOTHING was
        # destroyed — persisting from this instance would overwrite a good
        # row with an empty one, so _persist() refuses while this is set.
        self._load_failed = False
        # Dirty-tracking for _persist() — see that method's docstring.
        # Starts True so the first persist() call after construction always
        # writes; cleared only after a confirmed successful write.
        self._dirty = True
        # Exact serialized state last confirmed written, keyed by node id
        # and edge key. _persist() diffs the current graph against these
        # to write only the rows that actually changed, and to delete rows
        # for nodes/edges that no longer exist (prune() removes nodes and
        # rebuilds the edge list). Derived from real state rather than
        # from mutation bookkeeping, so no mutation site can forget to
        # register a change — see persistence.persist().
        self._persisted_nodes: dict[str, str] = {}
        self._persisted_edges: dict[str, str] = {}
        self._persisted_hashes_json: str = ""
        # True if this session's data came from the v1 blob table and is
        # awaiting its first per-row write.
        self._migrated_from_blob = False
        self._db_path = Path(storage_dir) / "graph_memory.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._safe_init_db()
        self._load()
        self._load_transitions()

    def _safe_init_db(self) -> None:
        """Initialize DB, deleting corrupt file if necessary. Full
        implementation in persistence.py."""
        from tokenmizer.graph_memory.persistence import safe_init_db
        safe_init_db(self)

    # ── DB ──────────────────────────────────────────────────────────────────
    # Full implementations in persistence.py.

    def _db_connect(self) -> "sqlite3.Connection":
        from tokenmizer.graph_memory.persistence import db_connect
        return db_connect(self)

    def _init_db(self) -> None:
        from tokenmizer.graph_memory.persistence import init_db
        init_db(self)

    def _persist_transition(self, t: DecisionTransition) -> None:
        from tokenmizer.graph_memory.persistence import persist_transition
        persist_transition(self, t)

    def get_transitions(self) -> list[DecisionTransition]:
        """Return all decision transitions for this session, newest first."""
        from tokenmizer.graph_memory.persistence import get_transitions
        return get_transitions(self)

    def _load_transitions(self) -> None:
        from tokenmizer.graph_memory.persistence import load_transitions
        load_transitions(self)

    def _persist(self, force: bool = False) -> bool:
        from tokenmizer.graph_memory.persistence import persist
        return persist(self, force=force)

    def _load(self) -> None:
        from tokenmizer.graph_memory.persistence import load
        load(self)

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
        source_role: str | None = "assistant",
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
                # NOTE: inactive (SUPERSEDED/INVALIDATED) nodes are
                # deliberately still eligible to absorb a match here.
                # Re-adding a *genuine restatement* of a dead decision
                # must be a no-op, not a revival: extraction re-scans old
                # messages whenever _processed_hashes is capped (see
                # extract_from_messages), so a stale "Use React" resurfacing
                # after the team moved to Next.js must not flip the choice
                # back. test_merge_does_not_resurrect_superseded covers this.
                #
                # A genuinely DIFFERENT choice in the same slot no longer
                # reaches this branch at all — _is_same_decision now
                # rejects competing alternatives outright, so it falls
                # through to node creation and the supersession path
                # rather than being swallowed by the dead node.
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
            source_role=source_role,
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

                # CONTESTED: decisions sharing a topic bucket with the
                # new one, but NOT confident enough to supersede (see
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
                # Non-fatal by design: a bug in contradiction detection
                # must not block creating the decision node itself.
                #
                # Warning, not debug: transition tracking ("Changed X → Y"
                # in resume context) could otherwise stop working entirely
                # and only be noticed as an absence. Also counted on the
                # instance and surfaced via stats(), so graph health is
                # inspectable without tailing logs.
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
            # Only heuristic-extracted decisions carry a real
            # source_role — HybridExtractor knows which message each came
            # from. LLM-synthesized decisions have no single-turn
            # attribution, so fall back to the same "assistant" default
            # every other node type uses. Passing an explicit None instead
            # would override that default and cost them the trust bonus
            # unattributed nodes otherwise get (measured: it regressed
            # acceptance).
            nid = self.add_node(NodeType.DECISION, label, NodeStatus.COMPLETED,
                                summary=summary, importance=0.9,
                                confidence=node_confidence,
                                source_role=d.get("source_role") or "assistant")
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

                # SUPERSEDES edges are deliberately NOT inferred from
                # word overlap. Linking a decision to every superseded
                # decision sharing a topic word invents causal history —
                # including links to decisions replaced by something else
                # entirely — and /why walks these edges to explain why a
                # choice is current, so invented edges become invented
                # explanations.
                #
                # add_node() creates the SUPERSEDES edge for each decision
                # this one genuinely replaced, alongside the
                # DecisionTransition that justifies it.

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
            if node.status in INACTIVE_STATUSES:
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
        """Build tiered resume context block for LLM injection. Full
        implementation in context_block.py."""
        from tokenmizer.graph_memory.context_block import to_context_block
        return to_context_block(self, token_budget=token_budget)

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """
        Counts exclude _evicted nodes and any edge touching one, so this
        agrees with every other consumer (query(), to_context_block(),
        both visualization.py exporters) rather than reporting figures
        /api/graph/{id}/viz contradicts.
        """
        from tokenmizer.core.dto import GraphStatsDTO
        live_nodes = [n for n in self._nodes.values() if not n._evicted]
        # Only edges whose BOTH endpoints are live: an edge pointing at
        # an evicted node is not rendered by /viz either.
        live_ids = {n.id for n in live_nodes}
        live_edges = [
            e for e in self._edges
            if e.source_id in live_ids and e.target_id in live_ids
        ]
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
            edge_count=len(live_edges),
            by_type=by_type,
            by_status=by_status,
            processed_messages=len(self._processed_hashes),
            avg_confidence=avg_confidence,
            decision_tracking_failures=self._decision_tracking_failures,
            persistence_broken=self._persistence_broken,
            data_loss_detected=self._data_loss_detected,
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
