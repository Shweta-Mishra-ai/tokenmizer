"""
TokenMizer Ontology — the formal vocabulary of the knowledge graph.

This module is the single machine-readable definition of WHAT the graph
can contain (node types), HOW things relate (edge types with domain/range
semantics), and WHICH lifecycle paths are legal (the status state machine).
Everything here mirrors the enums in types.py — this module adds the
*semantics* that code and clients can reason over:

  - `ontology_dict()`   → JSON-able ontology, served at GET /api/ontology,
                          consumable by MCP clients, docs, and the visualizer
  - `is_valid_transition(frm, to)` → status state-machine check, used by
                          reasoning.consistency_check() to flag anomalies

Design note: the ontology describes and audits; it does not gate writes.
Graph ingestion is heuristic and must stay permissive — rejecting a write
because of an unexpected transition would silently lose session facts.
Instead, reasoning.consistency_check() reports violations after the fact.
"""
from __future__ import annotations

from tokenmizer.graph_memory.types import EdgeType, NodeStatus, NodeType

ONTOLOGY_VERSION = "1.0"

# ── Node types ────────────────────────────────────────────────────────────────

NODE_TYPES: dict[NodeType, str] = {
    NodeType.GOAL:        "Top-level objective of the session; never decays.",
    NodeType.TASK:        "Unit of work with a lifecycle (pending → in_progress → completed/failed).",
    NodeType.DECISION:    "A choice that constrains future work; the only type that participates in supersession.",
    NodeType.FILE:        "Source file touched or discussed in the session.",
    NodeType.ERROR:       "Bug, exception, or failure encountered; completed = resolved.",
    NodeType.CONCEPT:     "Domain concept or idea referenced across turns.",
    NodeType.DEPENDENCY:  "External package or service the project relies on.",
    NodeType.API:         "External API integrated or discussed.",
    NodeType.PROJECT:     "Project or repository context marker.",
    NodeType.AGENT:       "AI agent or automation participating in the session.",
    NodeType.ENVIRONMENT: "Runtime environment fact (language version, OS, infra); never decays.",
    NodeType.TEST:        "Test file or test result.",
    NodeType.ENDPOINT:    "HTTP endpoint definition.",
    NodeType.SCHEMA:      "Data model or DB schema; never decays.",
}

# ── Edge types: (domain, range, semantics) ───────────────────────────────────
# domain/range are the *typical* endpoint types — descriptive, not enforced.

EDGE_TYPES: dict[EdgeType, dict] = {
    EdgeType.DEPENDS_ON: {
        "domain": [NodeType.TASK, NodeType.PROJECT],
        "range": [NodeType.DEPENDENCY, NodeType.TASK],
        "semantics": "Source cannot proceed/function without target.",
    },
    EdgeType.RELATED_TO: {
        "domain": list(NodeType),
        "range": list(NodeType),
        "semantics": "Generic association inferred from co-occurrence.",
    },
    EdgeType.IMPLEMENTS: {
        "domain": [NodeType.FILE, NodeType.TASK],
        "range": [NodeType.DECISION, NodeType.GOAL, NodeType.ENDPOINT],
        "semantics": "Source realizes/embodies the target.",
    },
    EdgeType.FIXES: {
        "domain": [NodeType.TASK, NodeType.FILE],
        "range": [NodeType.ERROR],
        "semantics": "Source resolves the target error.",
    },
    EdgeType.BLOCKS: {
        "domain": [NodeType.ERROR, NodeType.TASK],
        "range": [NodeType.TASK, NodeType.GOAL],
        "semantics": "Target cannot complete while source is unresolved.",
    },
    EdgeType.PART_OF: {
        "domain": [NodeType.TASK, NodeType.FILE, NodeType.ENDPOINT],
        "range": [NodeType.PROJECT, NodeType.GOAL],
        "semantics": "Compositional containment.",
    },
    EdgeType.SUPERSEDES: {
        "domain": [NodeType.DECISION],
        "range": [NodeType.DECISION],
        "semantics": "Source replaced target; target's valid_until is set. "
                     "Every SUPERSEDES edge should have a matching "
                     "DecisionTransition record with trigger/reason/evidence.",
    },
    EdgeType.CONFLICTS_WITH: {
        "domain": [NodeType.DECISION],
        "range": [NodeType.DECISION],
        "semantics": "Symmetric: both decisions share a topic but neither "
                     "was confidently identified as replacing the other — "
                     "both are marked CONTESTED rather than one being "
                     "silently marked SUPERSEDED on ambiguous evidence. "
                     "No DecisionTransition record — this is an unresolved "
                     "conflict, not a causal transition.",
    },
}

# ── Status state machine ─────────────────────────────────────────────────────
# Allowed transitions. MODIFIED is an alias of SUPERSEDED (same value) and is
# intentionally not listed separately.

STATUS_TRANSITIONS: dict[NodeStatus, set[NodeStatus]] = {
    NodeStatus.PENDING:     {NodeStatus.IN_PROGRESS, NodeStatus.COMPLETED,
                             NodeStatus.FAILED, NodeStatus.INVALIDATED},
    NodeStatus.IN_PROGRESS: {NodeStatus.COMPLETED, NodeStatus.FAILED,
                             NodeStatus.PENDING, NodeStatus.INVALIDATED},
    NodeStatus.COMPLETED:   {NodeStatus.SUPERSEDED, NodeStatus.INVALIDATED,
                             NodeStatus.IN_PROGRESS, NodeStatus.CONTESTED},
    NodeStatus.FAILED:      {NodeStatus.IN_PROGRESS, NodeStatus.COMPLETED,
                             NodeStatus.INVALIDATED},
    NodeStatus.SUPERSEDED:  {NodeStatus.ARCHIVED, NodeStatus.INVALIDATED},
    NodeStatus.INVALIDATED: set(),          # terminal — explicit human verdict
    NodeStatus.ARCHIVED:    set(),          # terminal — aged out
    # A contested decision is resolved by an explicit human/LLM
    # verdict — reasserting one side (back to COMPLETED), invalidating
    # one side, or a later decision genuinely superseding it. There is no
    # automatic resolution path.
    NodeStatus.CONTESTED:   {NodeStatus.COMPLETED, NodeStatus.SUPERSEDED,
                             NodeStatus.INVALIDATED},
}

STATUS_DESCRIPTIONS: dict[NodeStatus, str] = {
    NodeStatus.PENDING:     "Known but not started.",
    NodeStatus.IN_PROGRESS: "Actively being worked on.",
    NodeStatus.COMPLETED:   "Done / in effect. For decisions: the ACTIVE choice, shown in resume.",
    NodeStatus.FAILED:      "Attempted and failed; may be retried.",
    NodeStatus.SUPERSEDED:  "Replaced by a newer decision; kept in history, shown in resume ≤7 days.",
    NodeStatus.INVALIDATED: "Explicitly declared wrong/cancelled; always warned about in resume.",
    NodeStatus.ARCHIVED:    "Superseded >7 days ago; aged out of resume, retained in graph.",
    NodeStatus.CONTESTED:   "Shares a topic with another decision but neither confidently "
                            "replaces the other; both surfaced together, unresolved, in resume.",
}


def is_valid_transition(frm: NodeStatus, to: NodeStatus) -> bool:
    """True if the status state machine permits frm → to (self-loops allowed)."""
    if frm == to:
        return True
    return to in STATUS_TRANSITIONS.get(frm, set())


def ontology_dict() -> dict:
    """Full ontology as a JSON-serializable dict (served at /api/ontology)."""
    return {
        "version": ONTOLOGY_VERSION,
        "node_types": {t.value: desc for t, desc in NODE_TYPES.items()},
        "edge_types": {
            e.value: {
                "domain": [t.value for t in spec["domain"]],
                "range": [t.value for t in spec["range"]],
                "semantics": spec["semantics"],
            }
            for e, spec in EDGE_TYPES.items()
        },
        "statuses": {s.value: STATUS_DESCRIPTIONS[s]
                     for s in STATUS_DESCRIPTIONS},
        "status_transitions": {
            frm.value: sorted(t.value for t in tos)
            for frm, tos in STATUS_TRANSITIONS.items()
        },
    }
