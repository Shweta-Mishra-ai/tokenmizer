"""
Session / graph inspection, checkpoint, and decision-management endpoints.

Extracted from app.py to keep that file focused on the core proxy path
(POST /v1/chat/completions, /health, app setup). These ~15 auxiliary
REST endpoints (graph stats/viz/history/reasoning/obsidian exports,
manual checkpoints, decision invalidation, resume) are all read/inspect
or session-admin operations layered on top of the same GraphMemory
instances the proxy path manages — none of them are on the hot request
path, so isolating them here means a change to one doesn't require
re-reading (or risking a regression in) chat_completions() and vice
versa.

Shared state (settings, singletons like _analytics/_cache/_checkpoint_mgr,
and helpers like _get_graph_async/_check_rate_limit) stays defined in
app.py and is referenced here via `app_module.<name>` rather than
imported by value — this is the same lazy-module-reference pattern the
test suite already relies on (see e.g. test_persist_retry.py patching
`app_module._analytics`), so existing monkeypatch-based tests keep
working unchanged, and a value reassigned on app_module after import
(e.g. `monkeypatch.setattr(app_module, "_GRAPH_CACHE_MAX", ...)`) is
still honored since we never bind a stale local copy of it.

Pure code motion — no behavior changes.
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from tokenmizer.api import app as app_module
from tokenmizer.core.tokenizer import count_tokens
from tokenmizer.security.auth import verify_api_key
from tokenmizer.security.ownership import OwnershipUnavailable, SessionAccessDenied

logger = logging.getLogger(__name__)

router = APIRouter()


async def verify_session_access(request: Request) -> None:
    """FastAPI dependency enforcing session ownership on session-scoped
    routes. Must be listed AFTER verify_api_key, which establishes
    `request.state.principal`.

    Reads session_id from the path (e.g. /api/graph/{session_id}) or the
    query string (e.g. /api/checkpoint?session_id=...), whichever the
    route uses. A route with no session_id at all is unaffected.

    Read-only routes do not claim ownership (claim=False): a GET for a
    session that was never created should fall through to its normal
    404/empty response rather than staking a claim as a side effect.
    """
    session_id = (request.path_params.get("session_id")
                  or request.query_params.get("session_id"))
    if not session_id:
        return

    principal = getattr(request.state, "principal", None)
    if principal is None:
        # verify_api_key didn't run or didn't set one — fail closed.
        raise HTTPException(
            status_code=503,
            detail="Session access could not be evaluated — request rejected.",
        )

    claim = request.method not in ("GET", "HEAD", "OPTIONS")
    try:
        app_module._ownership.check_access(session_id, principal, claim=claim)
    except SessionAccessDenied:
        logger.warning(
            f"Denied {request.method} {request.url.path} — session "
            f"{session_id!r} belongs to a different principal"
        )
        # 404, not 403: confirming that a session exists but belongs to
        # someone else is itself a disclosure (it turns the endpoint into
        # a session-name oracle). Indistinguishable from "no such session".
        raise HTTPException(
            status_code=404,
            detail=f"No session '{session_id}' found.",
        )
    except OwnershipUnavailable as e:
        logger.error(f"Ownership store unavailable, denying request: {e}")
        raise HTTPException(
            status_code=503,
            detail="Session ownership state unavailable — request rejected "
                   "for safety. This is a server-side problem.",
        )


def _internal_error(context: str, e: Exception) -> HTTPException:
    """
    Build a 500 HTTPException for an unexpected failure without leaking
    the raw exception text to the client.

    these three handlers (checkpoint creation, decision
    invalidation, resume) must not put `detail=str(e)` in a response —
    `str(e)` on things like `sqlite3.OperationalError` or a filesystem
    error routinely embeds real disk paths, and other exception types
    can include similarly internal detail. `chat_completions()`'s own
    provider-failure handler already avoids this via a
    correlation id: full detail goes to the server log, the client gets
    a generic message plus the id to reference when asking for help.
    Factored out here since the same pattern was needed at all three
    call sites — one shared helper instead of three copies.
    """
    correlation_id = uuid.uuid4().hex[:12]
    logger.error(f"{context} [{correlation_id}]: {e}")
    return HTTPException(
        status_code=500,
        detail=f"{context} (ref: {correlation_id}). Check server logs for details.",
    )


@router.get("/api/stats", dependencies=[Depends(verify_api_key), Depends(verify_session_access), Depends(app_module._check_rate_limit)])
async def stats(session_id: Optional[str] = None):
    return app_module._analytics.summary()


class AnalyzeRequest(BaseModel):
    """File analysis request. Content is sent inline rather than as a
    path: the server may be a container or a remote host, so a path the
    CLIENT can see usually means nothing to it, and accepting one would
    be an arbitrary-file-read primitive besides."""
    filename: str
    content: str
    token_budget: int = 500
    query: str = ""


@router.post("/api/analyze", dependencies=[Depends(verify_api_key), Depends(app_module._check_rate_limit)])
async def analyze_file(req: AnalyzeRequest):
    """Summarise a large file into a token-budgeted digest.

    The same FileIntelligence used by layer 0 of the proxy pipeline and
    by the `analyze` plugin skill, exposed for callers that are not
    inside Claude Code — a shell, a script, curl, another editor. The
    README previously documented this as a known missing piece.
    """
    if req.token_budget <= 0 or req.token_budget > 100_000:
        raise HTTPException(
            status_code=422,
            detail="token_budget must be between 1 and 100000.",
        )
    if not req.filename.strip():
        raise HTTPException(status_code=422, detail="filename is required.")

    try:
        result = app_module._file_intelligence.process(
            req.content, req.filename,
            token_budget=req.token_budget, query=req.query,
        )
    except Exception as e:
        raise _internal_error("File analysis failed", e)

    return {
        "filename": req.filename,
        "file_type": result.file_type,
        "original_tokens": result.original_tokens,
        "extracted_tokens": result.extracted_tokens,
        "tokens_saved": result.tokens_saved,
        "savings_pct": result.savings_pct,
        "strategy_used": result.strategy_used,
        "was_truncated": result.was_truncated,
        "content": result.content,
        "summary": result.summary,
    }


@router.get("/api/cache/stats", dependencies=[Depends(verify_api_key), Depends(verify_session_access), Depends(app_module._check_rate_limit)])
async def cache_stats():
    # NOTE: no "preference_context" field is returned here.
    # SemanticCache._preference_store. PreferenceStore.save() has no
    # callers anywhere in the codebase, so that field was always the
    # empty string while implying a working cross-session preference-
    # memory feature. Reporting an always-empty field for an unwired
    # subsystem is worse than reporting nothing, so it is gone until the
    # store is actually populated — which additionally needs a decision
    # about scoping, since the store is process-global and would
    # otherwise share one caller's preferences with every other
    # principal (see security/ownership.py).
    return app_module._cache.stats()


@router.get("/api/graph/{session_id}/history", dependencies=[Depends(verify_api_key), Depends(verify_session_access), Depends(app_module._check_rate_limit)])
async def get_graph_history(session_id: str, at_time: float = 0.0, top_k: int = 12):
    """
    Query graph state at a specific Unix timestamp.
    at_time=0.0 (default) returns current state (equivalent to /viz).
    at_time=<unix_ts> returns which nodes were active at that point in time.

    Useful for: debugging decision changes, audit trail, "what did we decide
    at 2pm?" queries.
    """
    graph = await app_module._get_graph_async(session_id)
    if at_time == 0.0:
        nodes = graph.query("", top_k=top_k)
    else:
        nodes = graph.query_at_time("", at_time=at_time, top_k=top_k)
    return {
        "session_id": session_id,
        "at_time": at_time or None,
        "nodes": [
            {
                "id": n.id, "label": n.label, "type": n.type.value,
                "status": n.status.value, "importance": n.importance,
                "valid_from": n.valid_from, "valid_until": n.valid_until or None,
            }
            for n in nodes
        ],
        "count": len(nodes),
    }


@router.get("/api/graph/{session_id}", dependencies=[Depends(verify_api_key), Depends(verify_session_access), Depends(app_module._check_rate_limit)])
async def get_graph(session_id: str):
    graph = await app_module._get_graph_async(session_id)
    return graph.stats()


@router.get("/api/graph/{session_id}/viz", dependencies=[Depends(verify_api_key), Depends(verify_session_access), Depends(app_module._check_rate_limit)])
async def get_graph_viz(session_id: str):
    """
    Return full graph as D3-compatible JSON for visualization.
    {nodes: [...], edges: [...], meta: {...}}
    Used by the dashboard Graph tab and any external viz tool.
    """
    graph = await app_module._get_graph_async(session_id)
    return graph.to_vis_json()


@router.get("/api/graph/{session_id}/html", dependencies=[Depends(verify_api_key), Depends(verify_session_access), Depends(app_module._check_rate_limit)])
async def get_graph_html(session_id: str):
    """Shareable standalone interactive graph — open in a browser, drag/zoom,
    screenshot, share. Self-contained dark-theme D3 force layout."""
    from tokenmizer.graph_memory.visualization import to_share_html
    graph = await app_module._get_graph_async(session_id)
    return HTMLResponse(to_share_html(graph))


@router.get("/api/ontology", dependencies=[Depends(verify_api_key), Depends(verify_session_access), Depends(app_module._check_rate_limit)])
async def get_ontology():
    """The TokenMizer graph ontology: node/edge types with semantics and
    the status state machine. Machine-readable — what the graph CAN contain
    and which lifecycle transitions are legal."""
    from tokenmizer.graph_memory.ontology import ontology_dict
    return ontology_dict()


@router.get("/api/graph/{session_id}/why", dependencies=[Depends(verify_api_key), Depends(verify_session_access), Depends(app_module._check_rate_limit)])
async def get_graph_why(session_id: str, q: str):
    """Reasoning: trace the causal chain behind a decision. Matches decision
    nodes containing `q`, walks the supersession chain in both directions,
    and returns the old→new trail with trigger/reason/evidence per hop,
    plus the currently active choice."""
    from tokenmizer.graph_memory.reasoning import why
    graph = await app_module._get_graph_async(session_id)
    return why(graph, q)


@router.get("/api/graph/{session_id}/reasoning", dependencies=[Depends(verify_api_key), Depends(verify_session_access), Depends(app_module._check_rate_limit)])
async def get_graph_reasoning(session_id: str):
    """Full reasoning view over session memory: active decisions, recent
    changes, decision history grouped by topic, and an ontology-based
    consistency audit (contradictions, missing/dangling transitions)."""
    from tokenmizer.graph_memory.reasoning import summarize_reasoning
    graph = await app_module._get_graph_async(session_id)
    return summarize_reasoning(graph)


@router.get("/api/graph/{session_id}/obsidian", dependencies=[Depends(verify_api_key), Depends(verify_session_access), Depends(app_module._check_rate_limit)])
async def get_graph_obsidian(session_id: str):
    """
    Download graph as Obsidian Canvas (.canvas) file.
    Save as <any-name>.canvas inside your Obsidian vault and open directly.
    """
    import json as _json

    graph = await app_module._get_graph_async(session_id)
    canvas = graph.to_obsidian_canvas()
    filename = f"tokenmizer-{session_id[:12]}.canvas"
    return Response(
        content=_json.dumps(canvas, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/api/graph/{session_id}/transitions", dependencies=[Depends(verify_api_key), Depends(verify_session_access), Depends(app_module._check_rate_limit)])
async def get_transitions(session_id: str):
    """Full decision transition history — trigger, reason, evidence, confidence_delta."""
    graph = await app_module._get_graph_async(session_id)
    return {
        "session_id": session_id,
        "transitions": [
            {
                "id": t.id,
                "from_label": t.from_label,
                "to_label": t.to_label,
                "trigger": t.trigger,
                "reason": t.reason,
                "evidence": t.evidence,
                "confidence_delta": t.confidence_delta,
                "timestamp": t.timestamp,
                "context_line": t.to_context_line(),
            }
            for t in graph.get_transitions()
        ],
        "count": len(graph.get_transitions()),
    }


@router.post("/api/checkpoint", dependencies=[Depends(verify_api_key), Depends(verify_session_access), Depends(app_module._check_rate_limit)])
async def create_manual_checkpoint(session_id: str):
    """
    Create a manual checkpoint for a session, snapshotting current graph
    state. Used by `tokenmizer checkpoint <session-id>` (CLI) and the
    `/tokenmizer:checkpoint` Claude Code skill.

    FOUND DURING A FINAL ACCURACY PASS: this endpoint was referenced by
    the README's API Reference table, cli.py's `checkpoint` command, AND
    the Claude Code checkpoint skill (.claude-plugin/skills/checkpoint/
    SKILL.md) — all three call `POST /api/checkpoint?session_id=...` —
    but it was never actually implemented here. Every one of those three
    callers would have gotten a 404 against the real running app. This
    wasn't a documentation typo; it was a real, consistent gap across
    three independent consumers that nothing caught because none of them
    were exercised end-to-end during the original audit.

    Design note: unlike the auto-checkpoint path in chat_completions(),
    this has no live message history to extract from (a standalone HTTP
    call has no conversation attached) — `CheckpointManager.create()` is
    called with `messages=[]`, which is safe: extract_from_messages()
    early-returns on an empty new-messages diff, and the checkpoint still
    correctly snapshots whatever's ALREADY in the graph from prior chat
    turns. Verified with a direct test before writing this (see
    tests/unit/test_graph_persistence.py for the equivalent pattern).
    """
    try:
        graph = await app_module._get_graph_async(session_id)
        ckpt = app_module._checkpoint_mgr.create(
            session_id=session_id,
            messages=[],
            graph=graph,
            context_pct=0.0,
            trigger="manual",
        )
        return {
            "checkpoint_id": ckpt.checkpoint_id,
            "session_id": session_id,
            "node_count": len(ckpt.graph_snapshot.get("nodes", [])),
            "resume_tokens": ckpt.resume_tokens,
            "resume_standard": ckpt.resume_standard,
            "trigger": ckpt.trigger,
        }
    except Exception as e:
        raise _internal_error(f"Manual checkpoint failed for session {session_id}", e)


@router.get("/api/checkpoints/{session_id}", dependencies=[Depends(verify_api_key), Depends(verify_session_access), Depends(app_module._check_rate_limit)])
async def list_checkpoints(session_id: str):
    return app_module._checkpoint_mgr.list_checkpoints(session_id)


_INVALIDATE_MIN_LABEL_LEN = 3  # matches validator.py's own noise-pattern floor (<=3 chars = noise)


@router.post("/api/decision/invalidate", dependencies=[Depends(verify_api_key), Depends(verify_session_access), Depends(app_module._check_rate_limit)])
async def invalidate_decision(
    session_id: str,
    decision_label: Optional[str] = None,
    node_id: Optional[str] = None,
    reason: str = "",
):
    """
    Mark a decision as INVALIDATED (red) — explicitly wrong or cancelled.
    Use when a decision was made that turned out to be incorrect.
    History is preserved; decision is flagged as a warning in future resumes.

    Provide EITHER node_id (precise — targets exactly one node, e.g. an
    id copied from /api/graph/{session_id}/viz or /transitions) OR
    decision_label (fuzzy — word-boundary substring match against active
    decision labels; may match more than one node, all of which are
    returned in affected_nodes).

    decision_label must not be matched with a raw substring
    check (`label_lower in node.label.lower()`) and had no minimum
    length — `decision_label=""` is a substring of every label, so an
    empty (or accidentally-empty, e.g. a client bug that sends "") value
    invalidated EVERY active decision in the session in one call. A short
    label caused a milder version of the same problem: "sql" would match
    "PostgreSQL" AND "SQLAlchemy" AND any future "MySQL" decision as a
    side effect of literal substring containment, not because they're
    actually related. Fixed by requiring a minimum length and matching on
    a WORD-BOUNDARY substring instead of a raw one.
    """
    if node_id is None and (decision_label is None or
                            len(decision_label.strip()) < _INVALIDATE_MIN_LABEL_LEN):
        raise HTTPException(
            status_code=400,
            detail=(f"decision_label must be at least {_INVALIDATE_MIN_LABEL_LEN} "
                    f"characters (a shorter value matches too many unrelated "
                    f"decisions to safely invalidate) — or pass node_id for a "
                    f"precise, single-node match."),
        )
    try:
        import re

        from tokenmizer.graph_memory.graph import NodeStatus, NodeType
        graph = await app_module._get_graph_async(session_id)

        invalidated: list[dict] = []

        if node_id is not None:
            node = graph._nodes.get(node_id)
            if (node is not None and node.type == NodeType.DECISION
                    and node.status == NodeStatus.COMPLETED):
                node.status = NodeStatus.INVALIDATED
                node.summary = (
                    f"Invalidated: {reason[:100]}" if reason else "Explicitly invalidated"
                )
                invalidated.append({"node_id": node_id, "label": node.label})
        else:
            label_lower = decision_label.lower().strip()
            pattern = re.compile(r'\b' + re.escape(label_lower) + r'\b')
            # Only ACTIVE (COMPLETED) decisions are eligible: matching
            # across all statuses would let a label overwrite SUPERSEDED
            # history nodes with INVALIDATED, destroying their
            # supersession record. The response lists every affected node
            # so multi-matches are visible.
            for nid, node in graph._nodes.items():
                if (node.type == NodeType.DECISION and
                        node.status == NodeStatus.COMPLETED and
                        pattern.search(node.label.lower())):
                    node.status = NodeStatus.INVALIDATED
                    node.summary = (
                        f"Invalidated: {reason[:100]}" if reason else "Explicitly invalidated"
                    )
                    invalidated.append({"node_id": nid, "label": node.label})

        if not invalidated:
            target = node_id if node_id is not None else decision_label
            raise HTTPException(
                status_code=404,
                detail=(f"No ACTIVE decision matching '{target}' found in "
                        f"session '{session_id}' (superseded/archived decisions "
                        f"are not invalidatable — they are already inactive)")
            )
        # direct node mutation above bypasses add_node's dirty-tracking —
        # force=True is required here or this write is silently skipped
        # (caught in a final accuracy pass; same class of bug the
        # eviction path and prune() were already protected against).
        # _persist() now returns bool  — check it, since claiming
        # "status": "invalidated" while the write actually failed is the
        # same silent-data-loss pattern this whole audit is about.
        if not graph._persist(force=True):
            raise HTTPException(
                status_code=500,
                detail=(f"Decision(s) marked invalidated in memory, but the "
                        f"write to disk FAILED for session '{session_id}' — "
                        f"the change did not persist and will be lost on "
                        f"restart or cache eviction. Retry the request."),
            )
        return {
            "session_id": session_id,
            "invalidated": node_id if node_id is not None else decision_label,
            "affected_nodes": invalidated,
            "reason": reason,
            "status": "invalidated",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise _internal_error("Invalidate decision failed", e)


@router.get("/api/resume/{session_id}", dependencies=[Depends(verify_api_key), Depends(verify_session_access), Depends(app_module._check_rate_limit)])
async def get_resume(session_id: str, level: str = "standard"):
    """Get resume context for a session. level: critical | standard | full"""
    try:
        if level not in ("critical", "standard", "full"):
            level = "standard"
        ckpt = app_module._checkpoint_mgr.get_latest(session_id)
        if not ckpt:
            raise HTTPException(status_code=404, detail="No checkpoint found for session")
        resume_map = {
            "critical": ckpt.resume_critical,
            "standard": ckpt.resume_standard,
            "full": ckpt.resume_full,
        }
        text = resume_map.get(level, ckpt.resume_standard)
        return {
            "session_id": session_id,
            "checkpoint_id": ckpt.checkpoint_id,
            "level": level,
            "resume_context": text,
            "token_count": count_tokens(text),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise _internal_error(f"Resume failed for {session_id}", e)
