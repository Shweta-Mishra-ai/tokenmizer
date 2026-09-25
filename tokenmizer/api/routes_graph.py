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
from pydantic import BaseModel, Field

from tokenmizer.api import app as app_module
from tokenmizer.core.tokenizer import count_tokens
from tokenmizer.security.auth import verify_api_key
from tokenmizer.security.ownership import (
    DEV_PRINCIPAL,
    OwnershipUnavailable,
    SessionAccessDenied,
)
from tokenmizer.security.redaction import redact_messages

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
    # Preferences are NOT reported here. They used to live inside the
    # semantic cache and were never written, so this endpoint carried a
    # permanently-empty field implying a feature that did not run. They
    # are their own thing now, per principal, under /api/preferences.
    return app_module._cache.stats()


@router.get("/api/preferences", dependencies=[Depends(verify_api_key), Depends(app_module._check_rate_limit)])
async def list_preferences(request: Request):
    """What this principal's habits are remembered as, and what that costs.

    A memory of a person has to be readable by that person. `injected` is
    the exact text added to a system prompt, so "why does it keep doing
    that" has an answer you can look at rather than infer.
    """
    store = app_module._preferences
    if store is None:
        return {"enabled": False, "preferences": [], "injected": "",
                "note": "preferences.enabled is off; nothing is remembered "
                        "and nothing is injected."}
    principal = getattr(request.state, "principal", DEV_PRINCIPAL)
    settings = app_module.settings.preferences
    return {
        "enabled": True,
        "preferences": [{"key": k, "value": v}
                        for k, v in store.all(principal)],
        "injected": store.context(principal, max_items=settings.max_items,
                                  max_chars=settings.max_chars),
    }


@router.delete("/api/preferences", dependencies=[Depends(verify_api_key), Depends(app_module._check_rate_limit)])
async def forget_preferences(request: Request, key: str = ""):
    """Forget one preference, or all of them.

    A memory with no way to say "stop remembering that" is a liability.
    """
    store = app_module._preferences
    if store is None:
        return {"enabled": False, "forgotten": 0}
    principal = getattr(request.state, "principal", DEV_PRINCIPAL)
    return {"enabled": True, "forgotten": store.forget(principal, key or None)}


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
    # The header is Latin-1 by the HTTP spec, and the id is client-supplied:
    # a Hindi or emoji session id raised while the response was built (a
    # 500), and quotes or semicolons would reach the header. Same allowlist
    # as the lock-file name.
    safe = "".join(c if (c.isascii() and (c.isalnum() or c in "-_")) else "_"
                   for c in session_id[:12])
    filename = f"tokenmizer-{safe}.canvas"
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


class CheckpointBody(BaseModel):
    """Optional transcript to fold into the graph before snapshotting.

    A checkpoint made without one only snapshots what earlier chat turns
    already put in the graph. The MCP tool and the CLI have no such turns
    — they run beside the conversation, not through the proxy — so their
    checkpoints claimed the session, listed it on the dashboard, and left
    it at zero nodes. Passing the transcript here is how they populate it.
    Same message shape as the chat endpoint, capped so one call cannot
    ask the extractor to chew through an unbounded history.
    """
    messages: list[app_module.ChatMessage] = Field(default_factory=list, max_length=500)


@router.post("/api/checkpoint", dependencies=[Depends(verify_api_key), Depends(verify_session_access), Depends(app_module._check_rate_limit)])
async def create_manual_checkpoint(session_id: str, body: CheckpointBody | None = None):
    """
    Create a manual checkpoint for a session, snapshotting current graph
    state. Used by `tokenmizer checkpoint <session-id>` (CLI), the MCP
    `checkpoint_session` tool, and the `/tokenmizer:checkpoint` skill.

    FOUND DURING A FINAL ACCURACY PASS: this endpoint was referenced by
    the README's API Reference table, cli.py's `checkpoint` command, AND
    the Claude Code checkpoint skill (.claude-plugin/skills/checkpoint/
    SKILL.md) — all three call `POST /api/checkpoint?session_id=...` —
    but it was never actually implemented here. Every one of those three
    callers would have gotten a 404 against the real running app. This
    wasn't a documentation typo; it was a real, consistent gap across
    three independent consumers that nothing caught because none of them
    were exercised end-to-end during the original audit.

    With a `messages` body the transcript goes through the same steps the
    chat path applies before anything touches a graph — redaction at
    ingestion, then the per-session lock so this cannot interleave with
    a background extraction for the same session — and
    `CheckpointManager.create()` extracts from it. Without a body the
    call still snapshots whatever earlier chat turns already stored,
    which is what the pre-existing callers relied on.
    """
    try:
        graph = await app_module._get_graph_async(session_id)
        raw_messages = [
            {"role": m.role, "content": m.text()} for m in (body.messages if body else [])
        ]
        raw_messages = redact_messages(raw_messages)
        async with app_module._get_session_lock(session_id):
            ckpt = app_module._checkpoint_mgr.create(
                session_id=session_id,
                messages=raw_messages,
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


@router.get("/api/sessions", dependencies=[Depends(verify_api_key), Depends(app_module._check_rate_limit)])
async def list_sessions(request: Request):
    """The caller's sessions, with size and recency, newest first.

    Scoped by ownership: graph_memory.db holds every principal's sessions in
    one file, and listing from it would show one API key's sessions to
    another. So the list comes from the ownership store, and per-session
    detail is read only for ids the caller owns. This is what the dashboard
    renders — it used to show a hard-coded demo legend because nothing could
    tell it which sessions existed.
    """
    principal = getattr(request.state, "principal", None)
    if principal is None:
        raise HTTPException(status_code=503,
                            detail="Session access could not be evaluated — request rejected.")
    try:
        owned = app_module._ownership.sessions_for(principal)
    except OwnershipUnavailable as e:
        logger.error(f"Ownership store unavailable listing sessions: {e}")
        raise HTTPException(status_code=503,
                            detail="Session ownership state unavailable.")

    sessions = []
    for session_id in owned:
        graph = await app_module._get_graph_async(session_id)
        stats = graph.stats()
        sessions.append({
            "session_id": session_id,
            "node_count": stats.get("node_count", 0),
            "edge_count": stats.get("edge_count", 0),
            "by_type": stats.get("by_type", {}),
            "updated_at": max((n.updated_at for n in graph._nodes.values()), default=0.0),
            "graph_url": f"/api/graph/{session_id}/html",
        })
    sessions.sort(key=lambda s: s["updated_at"], reverse=True)
    return {"sessions": sessions, "count": len(sessions)}


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


def _live_resume(graph, level: str, next_action: str = "") -> str:
    """A resume block built from the graph as it is NOW, at the same three
    tiers a checkpoint stores. Same builders, so the two sources read
    identically to the client."""
    mgr = app_module._checkpoint_mgr
    if level == "critical":
        return mgr._build_critical(graph, next_action)
    if level == "full":
        return mgr._build_full(graph, [], next_action)
    return mgr._build_standard(graph, next_action)


@router.get("/api/resume/{session_id}", dependencies=[Depends(verify_api_key), Depends(verify_session_access), Depends(app_module._check_rate_limit)])
async def get_resume(session_id: str, level: str = "standard"):
    """Get resume context for a session. level: critical | standard | full

    Source of truth is the graph, not the checkpoint table. A checkpoint is
    a snapshot taken at one moment; the graph is persisted on every turn.
    Two cases used to return the wrong thing here:

    - No checkpoint at all -> 404, even when the session had a full graph.
      The auto-checkpoint trigger measures the request AFTER windowing
      (see api/app.py), and windowing keeps the request small, so on a
      default config a long proxy session often never crosses the
      threshold and never gets a checkpoint. "No checkpoint found" was
      then the answer to a session with hours of memory in it.
    - A checkpoint older than the graph -> the stale snapshot, with every
      decision made since silently missing.

    Now: if the graph has been updated since the latest checkpoint (or
    there is none), the block is built live from the graph and the
    response says `source: "live_graph"`. The checkpoint's own
    "Continue from" hint is kept when one exists, since the graph does
    not record the last request. 404 only when there is neither a
    checkpoint nor a single node.
    """
    try:
        if level not in ("critical", "standard", "full"):
            level = "standard"
        ckpt = app_module._checkpoint_mgr.get_latest(session_id)
        graph = await app_module._get_graph_async(session_id)
        live_nodes = [n for n in graph._nodes.values() if not n._evicted]
        graph_updated_at = max((n.updated_at for n in live_nodes), default=0.0)

        if not ckpt and not live_nodes:
            raise HTTPException(
                status_code=404,
                detail="No checkpoint found for session, and its graph memory is "
                       "empty — nothing has been recorded under this session_id yet.",
            )

        # Prefer the graph whenever it is newer than the snapshot. A one
        # second grace absorbs the extraction the checkpoint itself ran.
        use_live = live_nodes and (ckpt is None or graph_updated_at > ckpt.created_at + 1.0)
        if use_live:
            text = _live_resume(graph, level, ckpt.next_action if ckpt else "")
            source = "live_graph"
        else:
            resume_map = {
                "critical": ckpt.resume_critical,
                "standard": ckpt.resume_standard,
                "full": ckpt.resume_full,
            }
            text = resume_map.get(level, ckpt.resume_standard)
            source = "checkpoint"
        return {
            "session_id": session_id,
            "checkpoint_id": ckpt.checkpoint_id if ckpt else None,
            "source": source,
            "level": level,
            "resume_context": text,
            "token_count": count_tokens(text),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise _internal_error(f"Resume failed for {session_id}", e)
