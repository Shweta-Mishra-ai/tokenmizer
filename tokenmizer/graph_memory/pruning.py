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
reasoning behind each fix embedded in the comments below (Background:
decay idempotence fix, Background: prune-cap enforcement fix).
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


def _protected(node) -> bool:
    """A node pruning never moves, however full the graph is.

    The previous fallback pass made every node outside a short list
    eligible once the cap was exceeded, which included the open bug the
    session was in the middle of fixing and the task being worked on.
    Those are exactly what a resume must say. An over-full graph is a
    bounded cost; losing them is not.
    """
    if node.type in (NodeType.GOAL, NodeType.SCHEMA):
        return True
    if node.type == NodeType.DECISION and node.status in (
            NodeStatus.COMPLETED, NodeStatus.CONTESTED, NodeStatus.INVALIDATED,
            NodeStatus.PENDING, NodeStatus.IN_PROGRESS):
        return True
    if node.type == NodeType.ERROR and node.status == NodeStatus.FAILED:
        return True
    if node.type == NodeType.TASK and node.status == NodeStatus.IN_PROGRESS:
        return True
    if node.type == NodeType.SUMMARY:
        return True
    return False


_RESOLVED = (NodeStatus.COMPLETED, NodeStatus.FAILED, NodeStatus.ARCHIVED,
             NodeStatus.SUPERSEDED, NodeStatus.MODIFIED)


def prune(
    graph: "GraphMemory",
    max_nodes: int = 500,
    max_age_days: float = 60.0,
) -> int:
    """Bring the graph back under `max_nodes` by moving its least useful
    nodes to cold storage. Returns how many were moved.

    Candidates are every unprotected node, least important first; among
    equals, history (superseded decisions, resolved nodes not touched in
    `max_age_days`) goes before anything recent, then the oldest.
    Importance decides first because it already encodes what decay and
    resolution have learned: a 0.1 TODO nobody returned to is worth less
    than a 0.9 completed task.
    Protected nodes (see _protected) are never moved; if they alone exceed
    the cap the graph stays over it and says so.

    Nothing is deleted outright. Each node goes to the session's
    graph_pruned table first, from which /api/graph/{id}/pruned lists it
    and a restore puts it back; if that write fails, nothing is pruned.
    The old version deleted, and also treated an ACTIVE decision older
    than 60 days as prunable while its own warning said active decisions
    never are.
    """
    if len(graph._nodes) <= max_nodes:
        return 0
    to_prune = len(graph._nodes) - max_nodes
    now = time.time()
    cutoff = now - max_age_days * 86400

    tier1: list[tuple[float, str]] = []
    tier2: list[tuple[float, str]] = []
    for nid, node in graph._nodes.items():
        if _protected(node):
            continue
        if node.status in _RESOLVED and (
                node.updated_at < cutoff
                or (node.type == NodeType.DECISION
                    and node.status in (NodeStatus.SUPERSEDED, NodeStatus.MODIFIED,
                                        NodeStatus.ARCHIVED))):
            tier1.append((node.importance, 0, node.updated_at, nid))
        else:
            tier2.append((node.importance, 1, node.updated_at, nid))
    chosen = [nid for *_, nid in sorted(tier1 + tier2)[:to_prune]]

    if len(chosen) < to_prune:
        logger.warning(
            "prune() for session %s: %d node(s) over the cap of %d, but only %d "
            "can be moved — the rest are active decisions, open errors, work in "
            "progress, goals or schemas, which are never pruned. The graph stays "
            "over the cap until some of them resolve.",
            graph.session_id, to_prune, max_nodes, len(chosen))
    if not chosen:
        return 0

    from tokenmizer.graph_memory.persistence import store_pruned
    if not store_pruned(graph, [graph._nodes[nid] for nid in chosen]):
        # Could not keep them, so do not remove them: an oversized graph is
        # recoverable, a deleted node is not.
        return 0

    gone = set(chosen)
    for nid in chosen:
        del graph._nodes[nid]
    graph._edges = [e for e in graph._edges
                    if e.source_id not in gone and e.target_id not in gone]
    graph._persist(force=True)
    logger.info("Moved %d node(s) of session %s to cold storage", len(chosen), graph.session_id)
    return len(chosen)
