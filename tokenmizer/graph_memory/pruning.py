"""
Graph decay + pruning — importance aging and the hard node-count cap.

Extracted from graph.py to keep that file focused on core memory logic
(node/edge CRUD, extraction application, query). Follows the same split
pattern already established for visualization.py: module-level functions
taking `graph: GraphMemory` as the first argument, mutating its state
directly, with GraphMemory keeping a one-line delegating method so
existing callers (`graph.apply_importance_decay()`, `graph.prune()`)
are unaffected.

Pure code motion — no logic changes. See graph.py's git history for the
reasoning behind each fix embedded in the comments below (TM-07 for the
decay idempotence fix, TM-19 for the prune-cap enforcement fix).
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from tokenmizer.graph_memory.types import NodeStatus, NodeType

if TYPE_CHECKING:
    from tokenmizer.graph_memory.graph import GraphMemory

logger = logging.getLogger(__name__)


def apply_importance_decay(graph: "GraphMemory") -> dict[str, float]:
    """
    Time-based importance decay — runs automatically during extract_from_messages.

    Decay rules (all intentional):
    - COMPLETED tasks: decay 15% per day after 3 days (they're done — less relevant)
    - SUPERSEDED decisions: decay 30% per day (old dead branches)
    - ERROR nodes (resolved): decay 20% per day
    - ACTIVE decisions: NO decay (current choices always matter)
    - GOALS: NO decay (always relevant for resume context)
    - IN_PROGRESS tasks: slight decay 5% per day after 7 days (stale WIP)

    Min importance floor = 0.1 (never fully disappear from graph)
    Max decay per call = 50% of current value (prevents single-call wipeout)

    Returns: dict of {node_id: new_importance} for changed nodes
    """
    changed: dict[str, float] = {}

    # Age SUPERSEDED decisions into ARCHIVED after
    # ARCHIVE_SUPERSEDED_AFTER_DAYS, measured from valid_until (the
    # moment of supersession, not node creation). This is the only
    # code path that sets ARCHIVED.
    _now = time.time()
    for _nid, _node in graph._nodes.items():
        if (_node.type == NodeType.DECISION
                and _node.status == NodeStatus.SUPERSEDED
                and not _node._evicted
                and _node.valid_until > 0.0
                and (_now - _node.valid_until) / 86400.0
                    >= graph.ARCHIVE_SUPERSEDED_AFTER_DAYS):
            _node.status = NodeStatus.ARCHIVED
            graph._dirty = True
            logger.debug(f"Archived long-superseded decision {_nid}: "
                         f"{_node.label[:60]}")

    # Decay rates per day
    _DECAY_RATE = {
        # (status, type): daily_decay_fraction
        (NodeStatus.COMPLETED,  NodeType.TASK):       0.15,
        (NodeStatus.COMPLETED,  NodeType.ERROR):      0.20,
        (NodeStatus.SUPERSEDED, NodeType.DECISION):   0.30,
        (NodeStatus.ARCHIVED,   NodeType.DECISION):   0.25,
        (NodeStatus.FAILED,     NodeType.TASK):       0.10,
        (NodeStatus.IN_PROGRESS, NodeType.TASK):      0.05,
    }
    _NO_DECAY_TYPES = {NodeType.GOAL, NodeType.ENVIRONMENT, NodeType.SCHEMA}
    _NO_DECAY_STATUSES = {NodeStatus.IN_PROGRESS, NodeStatus.PENDING}

    for nid, node in graph._nodes.items():
        if node._evicted:
            continue
        # Never decay goals, environments, schemas
        if node.type in _NO_DECAY_TYPES:
            continue
        # Never decay active decisions
        if node.type == NodeType.DECISION and node.status == NodeStatus.COMPLETED:
            continue

        rate = _DECAY_RATE.get((node.status, node.type), 0.0)
        if rate == 0.0:
            # Not currently in a decaying (status, type) bucket — keep
            # last_decayed_at current so a LATER transition into a
            # decaying bucket (e.g. IN_PROGRESS -> COMPLETED) isn't
            # charged decay for time spent in a non-decaying state.
            node.last_decayed_at = _now
            continue

        age_days = node.age_days()

        # Grace period: no decay in first N days of the node's life.
        # Governs whether decay should START at all — still based on
        # absolute age (via updated_at), separate from the elapsed-
        # since-last-decay measure below that governs how MUCH decay
        # applies once it's started.
        grace = {
            NodeType.TASK:  3.0,
            NodeType.ERROR: 1.0,
        }.get(node.type, 0.0)

        if age_days <= grace:
            node.last_decayed_at = _now
            continue

        # Decay magnitude: elapsed time SINCE THIS NODE'S LAST DECAY
        # APPLICATION, not absolute age. This is what makes repeated
        # calls (once per chat turn) idempotent per unit of real time
        # instead of compounding per call — see the field docstring
        # on MemoryNode.last_decayed_at.
        elapsed_days = max(0.0, (_now - node.last_decayed_at) / 86400.0)
        if elapsed_days <= 0.0:
            continue  # called again with no real time elapsed — nothing to do

        decay_factor = max(0.5, (1.0 - rate) ** elapsed_days)
        new_importance = max(0.10, round(node.importance * decay_factor, 3))

        if abs(new_importance - node.importance) > 0.005:
            node.importance = new_importance
            changed[nid] = new_importance
            graph._dirty = True
        node.last_decayed_at = _now

    return changed


def prune(
    graph: "GraphMemory",
    max_nodes: int = 200,
    max_age_days: float = 60.0,
) -> int:
    """Remove low-importance, old, completed nodes. Preserve decisions, envs, goals."""
    preserve_types = {NodeType.GOAL, NodeType.SCHEMA}
    # Decisions are kept even when old — history matters
    # But ARCHIVED/SUPERSEDED decisions can be pruned after max_age_days
    cutoff = time.time() - max_age_days * 86400
    # Superseded decisions expire faster (30 days default)
    superseded_cutoff = time.time() - min(max_age_days, 30) * 86400
    candidates: list[tuple[float, str]] = []

    for nid, node in graph._nodes.items():
        if node.type in preserve_types:
            continue
        # ACTIVE decisions and environments: keep unless very old
        if node.type in (NodeType.DECISION, NodeType.ENVIRONMENT):
            if node.status == NodeStatus.COMPLETED and node.updated_at < cutoff:
                score = node.importance * 0.1  # low score = prune first
                candidates.append((score, nid))
            elif node.status in (NodeStatus.SUPERSEDED, NodeStatus.MODIFIED,
                                 NodeStatus.ARCHIVED) and node.updated_at < superseded_cutoff:
                candidates.append((0.0, nid))  # prune superseded decisions after 30d
            continue
        # All other nodes: prune if old and completed
        if node.status in (NodeStatus.COMPLETED, NodeStatus.FAILED,
                           NodeStatus.ARCHIVED) and node.updated_at < cutoff:
            score = node.importance * (node.updated_at / (time.time() + 1))
            candidates.append((score, nid))

    if len(graph._nodes) <= max_nodes:
        return 0

    to_prune = len(graph._nodes) - max_nodes

    # If age-based pruning didn't find enough candidates (graph is fresh —
    # all nodes created recently), fall back to importance-only pruning.
    # This ensures the hard cap is always enforced even in long single-day sessions.
    #
    # FIXED (TM-19): this used to blanket-exclude EVERY node of type
    # DECISION, including SUPERSEDED/ARCHIVED/INVALIDATED ones. Only
    # ACTIVE (COMPLETED) decisions are meant to be permanently
    # protected — a graph dominated by dead/historical decisions
    # (all created within max_age_days, so the age-based scan above
    # found nothing) could never be pruned down to max_nodes at all,
    # making the "hard cap" not actually hard. Only ACTIVE decisions
    # (and GOAL/SCHEMA, via preserve_types) are excluded here now.
    if len(candidates) < to_prune:
        candidate_ids = {nid for _, nid in candidates}  # computed ONCE, not per iteration
        importance_candidates = [
            (node.importance, nid)
            for nid, node in graph._nodes.items()
            if node.type not in preserve_types
            and not (node.type == NodeType.DECISION and node.status == NodeStatus.COMPLETED)
            and nid not in candidate_ids
        ]
        candidates.extend(importance_candidates)

    # Sort ONCE, after every tier has been combined.
    #
    # FIXED (TM-19): this used to sort BEFORE the fallback tier's
    # extend() above, so the two tiers' entries were never merged
    # into one true ascending-score order — candidates[:to_prune]
    # could prune "all age-based candidates first regardless of
    # score, then fallback candidates appended in their own separate
    # order," rather than truly lowest-scoring-first across both.
    candidates.sort()

    if len(candidates) < to_prune:
        logger.warning(
            f"prune() for session {graph.session_id}: only found "
            f"{len(candidates)} evictable node(s) but need to remove "
            f"{to_prune} to reach max_nodes={max_nodes} — the graph is "
            f"dominated by protected node types (active decisions, "
            f"goals, schemas). Pruning what's available; the cap will "
            f"remain exceeded until more nodes become eligible. Active "
            f"decisions are intentionally never auto-pruned regardless "
            f"of cap pressure — deleting a current choice is a bigger "
            f"risk than a temporarily oversized graph."
        )

    pruned = 0

    for _, nid in candidates[:to_prune]:
        del graph._nodes[nid]
        graph._edges = [e for e in graph._edges
                       if e.source_id != nid and e.target_id != nid]
        pruned += 1

    if pruned:
        graph._persist(force=True)
        logger.info(f"Graph pruned {pruned} nodes for session {graph.session_id}")

    return pruned
