"""
Graph Reasoning — inference over session memory, not just storage.

The graph stores facts (nodes), relations (edges), and causal history
(DecisionTransitions). This module answers questions OVER that structure:

  why(graph, query)          "Why is X the current choice?" — walks the
                             supersession chain and returns the full
                             old→new trail with triggers/reasons/evidence.
  impact(graph, query)       "What is connected to X?" — typed 1-hop
                             neighborhood (files a decision touches, tasks
                             an error blocks, ...).
  decision_history(graph)    Timeline of decisions grouped by topic bucket.
  consistency_check(graph)   Ontology-based audit: contradictions the
                             tracker missed, superseded decisions with no
                             transition record, dangling transition
                             endpoints, illegal status states.
  summarize_reasoning(graph) One dict combining the above — the payload
                             behind GET /api/graph/{session}/reasoning.

Everything here is read-only and deterministic — no LLM calls, no writes.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from tokenmizer.graph_memory.decision_tracker import classify_topics
from tokenmizer.graph_memory.ontology import (
    is_valid_transition,  # noqa: F401  (re-export for callers)
)
from tokenmizer.graph_memory.types import (
    DecisionTransition,
    MemoryNode,
    NodeStatus,
    NodeType,
)

if TYPE_CHECKING:
    from tokenmizer.graph_memory.graph import GraphMemory

_INACTIVE = {NodeStatus.SUPERSEDED, NodeStatus.ARCHIVED, NodeStatus.INVALIDATED}


def _node_brief(n: MemoryNode) -> dict:
    return {
        "id": n.id,
        "type": n.type.value,
        "label": n.label,
        "status": n.status.value,
        "summary": n.summary,
        "confidence": n.confidence,
    }


def _transition_brief(t: DecisionTransition) -> dict:
    return {
        "from_id": t.from_decision_id,
        "to_id": t.to_decision_id,
        "from_label": t.from_label,
        "to_label": t.to_label,
        "trigger": t.trigger,
        "reason": t.reason,
        "evidence": t.evidence,
        "timestamp": t.timestamp,
    }


def why(graph: "GraphMemory", query: str) -> dict:
    """
    Trace the causal chain behind a decision.

    Matches decision nodes whose label contains `query` (case-insensitive),
    then walks the transition graph in BOTH directions (what this decision
    replaced, and what replaced it) until the chain ends. The result reads
    as a story: earliest choice → ... → current active choice, each hop
    carrying its trigger/reason/evidence.
    """
    q = (query or "").lower().strip()
    if not q:
        return {"query": query, "matches": [], "chain": [], "current": None}

    matched = [n for n in graph._nodes.values()
               if n.type == NodeType.DECISION and q in n.label.lower()]

    transitions = graph.get_transitions()
    # One decision can be superseded once, but a NEW decision may supersede
    # several old ones at once (multi-topic contradiction) — so both maps
    # must hold lists, not single transitions.
    by_from: dict[str, list[DecisionTransition]] = {}
    by_to: dict[str, list[DecisionTransition]] = {}
    for t in transitions:
        by_from.setdefault(t.from_decision_id, []).append(t)
        by_to.setdefault(t.to_decision_id, []).append(t)

    # Collect the full chain closure around every matched node
    chain: list[DecisionTransition] = []
    seen_ids: set[str] = set()
    frontier = [n.id for n in matched]
    visited: set[str] = set()
    while frontier:
        nid = frontier.pop()
        if nid in visited:
            continue
        visited.add(nid)
        for t in by_from.get(nid, []) + by_to.get(nid, []):
            if t.id in seen_ids:
                continue
            seen_ids.add(t.id)
            chain.append(t)
            frontier.extend([t.from_decision_id, t.to_decision_id])

    chain.sort(key=lambda t: t.timestamp)

    # The "current" answer: the active decision at the end of the chain,
    # or the matched node itself if it never transitioned.
    current = None
    endpoint_ids = ({t.to_decision_id for t in chain} - {t.from_decision_id for t in chain})
    for nid in endpoint_ids:
        node = graph._nodes.get(nid)
        if node is not None and node.status not in _INACTIVE:
            current = _node_brief(node)
            break
    if current is None:
        for n in matched:
            if n.status not in _INACTIVE:
                current = _node_brief(n)
                break

    return {
        "query": query,
        "matches": [_node_brief(n) for n in matched],
        "chain": [_transition_brief(t) for t in chain],
        "current": current,
    }


def impact(graph: "GraphMemory", query: str) -> dict:
    """
    Typed 1-hop neighborhood of every node matching `query` — which files,
    tasks, errors, dependencies are connected, and via which relation.
    """
    q = (query or "").lower().strip()
    matched = [n for n in graph._nodes.values() if q and q in n.label.lower()]
    matched_ids = {n.id for n in matched}

    connections = []
    for e in graph._edges:
        if e.source_id in matched_ids or e.target_id in matched_ids:
            src = graph._nodes.get(e.source_id)
            tgt = graph._nodes.get(e.target_id)
            if src is None or tgt is None:
                continue
            connections.append({
                "relation": e.type.value,
                "source": _node_brief(src),
                "target": _node_brief(tgt),
            })

    return {
        "query": query,
        "matches": [_node_brief(n) for n in matched],
        "connections": connections,
    }


def decision_history(graph: "GraphMemory") -> dict:
    """Decisions grouped by topic bucket, each bucket ordered oldest→newest."""
    buckets: dict[str, list[dict]] = {}
    for n in sorted(
        (n for n in graph._nodes.values() if n.type == NodeType.DECISION),
        key=lambda n: n.valid_from,
    ):
        topics = classify_topics(n.label, n.summary) or {"(unclassified)"}
        entry = _node_brief(n)
        entry["decided_at"] = n.valid_from
        entry["superseded_at"] = n.valid_until or None
        for topic in sorted(topics):
            buckets.setdefault(topic, []).append(entry)
    return buckets


def consistency_check(graph: "GraphMemory") -> list[dict]:
    """
    Ontology-based audit of the graph's current state. Returns a list of
    anomalies (empty list = consistent). Each anomaly: {kind, detail, ids}.
    """
    issues: list[dict] = []
    transitions = graph.get_transitions()
    transitioned_from_ids = {t.from_decision_id for t in transitions}

    # 1. Two ACTIVE decisions sharing a topic — a contradiction the
    #    supersession tracker failed to catch (e.g. vocabulary gap).
    active = [n for n in graph._nodes.values()
              if n.type == NodeType.DECISION and n.status == NodeStatus.COMPLETED]
    for i, a in enumerate(active):
        ta = classify_topics(a.label, a.summary)
        if not ta:
            continue
        for b in active[i + 1:]:
            shared = ta & classify_topics(b.label, b.summary)
            if shared:
                issues.append({
                    "kind": "active_contradiction",
                    "detail": (f"Two active decisions share topic(s) "
                               f"{sorted(shared)}: {a.label!r} vs {b.label!r}"),
                    "ids": [a.id, b.id],
                })

    # 2. SUPERSEDED decision with no transition record — history lost.
    for n in graph._nodes.values():
        if (n.type == NodeType.DECISION
                and n.status == NodeStatus.SUPERSEDED
                and n.id not in transitioned_from_ids):
            issues.append({
                "kind": "missing_transition",
                "detail": f"Decision {n.label!r} is SUPERSEDED but has no "
                          f"transition record explaining what replaced it",
                "ids": [n.id],
            })

    # 3. Transition endpoints missing from the graph (over-pruned).
    node_ids = set(graph._nodes.keys())
    for t in transitions:
        missing = [i for i in (t.from_decision_id, t.to_decision_id)
                   if i not in node_ids]
        if missing:
            issues.append({
                "kind": "dangling_transition",
                "detail": f"Transition {t.from_label!r} → {t.to_label!r} "
                          f"references node(s) no longer in the graph",
                "ids": missing,
            })

    return issues


def summarize_reasoning(graph: "GraphMemory") -> dict:
    """The full reasoning view — served at GET /api/graph/{session}/reasoning."""
    now = time.time()
    active_decisions = [
        _node_brief(n) for n in graph._nodes.values()
        if n.type == NodeType.DECISION and n.status == NodeStatus.COMPLETED
    ]
    recent_changes = [
        _transition_brief(t) for t in graph.get_transitions()
        if now - t.timestamp <= 7 * 86400
    ]
    return {
        "session_id": graph.session_id,
        "active_decisions": active_decisions,
        "recent_changes": recent_changes,
        "history_by_topic": decision_history(graph),
        "consistency": consistency_check(graph),
    }
