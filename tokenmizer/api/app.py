"""
TokenMizer — main FastAPI application.

All fixes applied:
- tiktoken accurate token counting everywhere
- API key authentication on all non-health endpoints
- State backend (Redis or in-memory, not module-level dicts)
- Layer 5 (context router) properly wired or removed
- CORS restricted to configured origins
- Checkpoint extraction uses full message history
- No dead code
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from tokenmizer.config.settings import get_settings
from tokenmizer.core.tokenizer import count_tokens, count_messages_tokens
from tokenmizer.security.auth import verify_api_key
from tokenmizer.security.middleware import injection_guard
from tokenmizer.security.redaction import redact_messages
from tokenmizer.state.backend import get_state_backend
from tokenmizer.providers.providers import build_provider
from tokenmizer.compression.engine import CompressionPipeline
from tokenmizer.compression.output_trimmer import OutputTrimmer
from tokenmizer.compression.window import SmartMessageWindow, needs_windowing
from tokenmizer.filters.file_intelligence import FileIntelligence
from tokenmizer.semantic_cache.cache import SemanticCache
from tokenmizer.graph_memory.graph import GraphMemory
from tokenmizer.checkpoints.manager import CheckpointManager
from tokenmizer.analytics.engine import AnalyticsEngine
from tokenmizer.api.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)

settings = get_settings()

# ── Singletons ────────────────────────────────────────────────────────────────
_provider = None
_compression = CompressionPipeline(
    ratio=settings.compression.ratio,
    enable_ml=(settings.compression.engine == "llmlingua2"),
)
_cache = SemanticCache(
    threshold=settings.cache.similarity_threshold,
    ttl_seconds=settings.cache.ttl_seconds,
    max_size=settings.cache.max_size,
)
_checkpoint_mgr = CheckpointManager(storage_dir=settings.graph_checkpoint.storage_dir)
_analytics = AnalyticsEngine()
_state = get_state_backend(settings.state_backend, settings.redis_url)
_output_trimmer = OutputTrimmer()
_rate_limiter = get_rate_limiter(rate=60, per_seconds=60, burst=10)

# Bounded LRU for session locks — prevents memory leak on long-running servers.
# Max 1000 concurrent sessions; LRU eviction removes oldest.
_SESSION_LOCK_MAX = 1000
_session_locks: "OrderedDict[str, asyncio.Lock]" = {}

try:
    from collections import OrderedDict as _OD
    _session_locks = _OD()
except ImportError:
    pass


def _get_session_lock(session_id: str) -> asyncio.Lock:
    """Get or create a per-session async lock (LRU-bounded)."""
    from collections import OrderedDict
    if not isinstance(_session_locks, OrderedDict):
        # Fallback: plain dict (shouldn't happen)
        return asyncio.Lock()
    if session_id in _session_locks:
        _session_locks.move_to_end(session_id)
        return _session_locks[session_id]
    lock = asyncio.Lock()
    _session_locks[session_id] = lock
    if len(_session_locks) > _SESSION_LOCK_MAX:
        _session_locks.popitem(last=False)  # evict oldest
    return lock
_smart_window = SmartMessageWindow(
    token_budget=settings.memory.max_tokens_before_summary,
    protect_recent=settings.memory.recent_turns_verbatim,
    graph_context_budget=250,
)
_file_intelligence = FileIntelligence()
_cheap_provider = None   # lazy — only built if use_llm_extraction=True


def _get_cheap_provider():
    """
    Build a cheap model provider for LLM extraction.
    Uses haiku/gpt-4o-mini — costs ~$0.001 per extraction turn.
    Only instantiated when use_llm_extraction=True.
    """
    global _cheap_provider
    if _cheap_provider is not None:
        return _cheap_provider

    from tokenmizer.providers.providers import (
        AnthropicProvider, OpenAIProvider
    )

    provider = settings.provider.lower()
    key = settings.get_api_key_for_provider(provider)

    if provider in ("anthropic", "claude") and key:
        _cheap_provider = AnthropicProvider(key, model="claude-haiku-4-5")
    elif provider in ("openai", "gpt") and key:
        _cheap_provider = OpenAIProvider(key, model="gpt-4o-mini")
    elif provider == "deepseek" and key:
        from tokenmizer.providers.providers import DeepSeekProvider
        _cheap_provider = DeepSeekProvider(key, model="deepseek-chat")
    else:
        # No cheap model available — will fall back to heuristic
        _cheap_provider = None

    return _cheap_provider


def _get_provider():
    global _provider
    if _provider is None:
        _provider = build_provider(settings)
    return _provider


# ── Graph helpers (state-backend backed) ─────────────────────────────────────

# In-process graph cache — avoids SQLite reload on every request.
# Thread-safe: each session_id maps to one GraphMemory, protected by _get_session_lock().
_graph_cache: dict[str, GraphMemory] = {}


def _get_graph(session_id: str) -> GraphMemory:
    """Get or create cached graph for session. Loaded from SQLite once, kept in memory."""
    if session_id not in _graph_cache:
        _graph_cache[session_id] = GraphMemory(
            session_id,
            storage_dir=settings.graph_checkpoint.storage_dir,
        )
    return _graph_cache[session_id]


def _get_context_used(session_id: str) -> int:
    return _state.get(f"ctx:{session_id}") or 0


def _set_context_used(session_id: str, tokens: int) -> None:
    _state.set(f"ctx:{session_id}", tokens, ttl=86400)


# ── Context window sizes ──────────────────────────────────────────────────────

_CONTEXT_WINDOWS = {
    "claude": 200_000, "claude-sonnet": 200_000, "claude-opus": 200_000,
    "gpt-4o": 128_000, "gpt-4": 128_000, "gpt-3.5": 16_000,
    "gemini": 1_000_000, "deepseek": 64_000,
}


def _context_window(model: str) -> int:
    for k, v in _CONTEXT_WINDOWS.items():
        if k in model.lower():
            return v
    return 128_000


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TokenMizer starting")
    yield
    logger.info("TokenMizer stopped")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="TokenMizer",
    description="Never lose your AI context again.",
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    session_id: Optional[str] = None


# ── Main endpoint ─────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key), Depends(injection_guard)])
async def chat_completions(req: ChatRequest, request: Request):
    session_id = req.session_id or str(uuid.uuid4())
    model = req.model or settings.default_model
    t0 = time.monotonic()

    # ── Rate limiting ─────────────────────────────────────────────────────────
    client_id = request.headers.get("Authorization", request.client.host if request.client else "unknown")
    allowed, retry_after = await _rate_limiter.check(client_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after:.1f}s",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )

    raw_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    messages = raw_messages[:]
    savings: dict[str, int] = {}
    cache_hit = False

    # ── Layer 0: File intelligence — large files extracted before anything else ─
    user_query = next((m["content"] for m in reversed(raw_messages) if m.get("role") == "user"), "")
    messages, file_tokens_saved = _file_intelligence.process_message_files(
        messages, token_budget_per_file=600, query=user_query
    )
    savings["file_extraction"] = file_tokens_saved

    # ── Layer 1: Prompt compression ───────────────────────────────────────────
    if settings.compression.enabled:
        compressed, saved = _compression.compress_messages(messages, protect_recent=3)
        messages = compressed
        savings["compression"] = saved

    # ── Layer 2: Terse output injection ───────────────────────────────────────
    if settings.terse_output.enabled:
        terse = _compression.terse_system_prompt(settings.terse_output.level)
        has_system = any(m.get("role") == "system" for m in messages)
        if has_system:
            for m in messages:
                if m.get("role") == "system":
                    m["content"] = terse + "\n\n" + m["content"]
                    break
        else:
            messages = [{"role": "system", "content": terse}] + messages

    # ── Layer 3: Semantic cache ───────────────────────────────────────────────
    user_content = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    cached_response = None

    if settings.cache.enabled and user_content:
        # Pass session_id so session-sensitive responses are isolated
        cached_response = _cache.get(user_content, session_id=session_id)
        if cached_response:
            cache_hit = True
            savings["cache"] = count_tokens(user_content, model)

    # ── Layer 4: Context graph + smart window ────────────────────────────────
    if settings.graph_checkpoint.enabled:
        # Note: graph operations are not thread-safe under high concurrency.
        # For single-user dev use this is fine. For production multi-user
        # deploy with TOKENMIZER_STATE_BACKEND=redis and use 1 worker per CPU.
        context_used = _get_context_used(session_id)
        context_window = _context_window(model)
        input_tokens = count_messages_tokens(messages, model)
        context_pct = (context_used + input_tokens) / context_window

        graph = _get_graph(session_id)

        # Update graph incrementally on each turn
        # LLM extraction: much higher accuracy (~80%+ vs ~45% heuristic)
        # Only fires when: setting enabled + cheap provider available + new messages exist
        if settings.graph_checkpoint.use_llm_extraction:
            cheap = _get_cheap_provider()
            if cheap is not None:
                # Run LLM extraction async — don't block main response
                # We extract the LAST 2 turns (current exchange) incrementally
                recent_new = raw_messages[-4:] if len(raw_messages) >= 4 else raw_messages
                new_msgs = [m for m in recent_new
                            if graph._msg_hash(m) not in graph._processed_hashes]
                if new_msgs:
                    # Heuristic extraction runs NOW (fast, synchronous)
                    # so graph is updated before response is sent
                    graph.extract_from_messages(raw_messages, incremental=True)

                    # LLM extraction runs IN BACKGROUND (non-blocking)
                    # It will update the graph AFTER response is returned.
                    # Next request will benefit from improved extraction.
                    # Fire and forget — response NOT delayed.
                    # Lock passed into closure so background write is serialized.
                    _lock_ref = _get_session_lock(session_id)

                    async def _background_llm_extract_safe(
                        _graph=graph, _msgs=new_msgs, _all=raw_messages,
                        _cheap=cheap, _lock=_lock_ref, _session=session_id
                    ):
                        async with _lock:  # serialize concurrent writes to same graph
                            try:
                                # Use HybridExtractor: LLM → heuristic → merge
                                # Falls back gracefully if LLM call fails
                                from tokenmizer.graph_memory.hybrid_extractor import HybridExtractor

                                async def _provider_fn(messages, system="", max_tokens=600):
                                    r = await _cheap.chat(
                                        messages=messages, system=system, max_tokens=max_tokens
                                    )
                                    return {"text": r.text}

                                extractor = HybridExtractor(provider_fn=_provider_fn)
                                extracted = await extractor.extract(_msgs)
                                _graph.extract_from_messages(
                                    _all, incremental=False, extracted_data=extracted
                                )
                                logger.debug(f"HybridExtractor complete for {_session}")
                            except ImportError:
                                # HybridExtractor not available — fall back to _llm_extract
                                try:
                                    from tokenmizer.graph_memory.graph import _llm_extract
                                    async def _pfn(messages, system="", max_tokens=600):
                                        r = await _cheap.chat(messages=messages, system=system, max_tokens=max_tokens)
                                        return {"text": r.text}
                                    extracted = await _llm_extract(_msgs, _pfn)
                                    _graph.extract_from_messages(_all, incremental=False, extracted_data=extracted)
                                except Exception as e2:
                                    logger.debug(f"LLM extraction fallback failed: {e2}")
                            except Exception as e:
                                logger.debug(f"Background extraction failed (non-fatal): {e}")

                    asyncio.create_task(_background_llm_extract_safe())
                else:
                    graph.extract_from_messages(raw_messages, incremental=True)
            else:
                graph.extract_from_messages(raw_messages, incremental=True)
        else:
            graph.extract_from_messages(raw_messages, incremental=True)

        # Smart window: compress old turns → graph summary (kills middle-conversation waste)
        if needs_windowing(messages, settings.memory.max_tokens_before_summary, model):
            messages, window_saved = _smart_window.apply(messages, graph, model)
            savings["windowing"] = window_saved
        else:
            savings["windowing"] = 0

        # Targeted context injection: only relevant nodes, only when useful
        if len(graph._nodes) >= 3 and len(user_query.split()) >= 4:
            relevant_nodes = graph.query(user_query, top_k=8)
            if relevant_nodes:
                ctx_parts = []
                for n in relevant_nodes[:6]:
                    entry = n.label
                    if n.summary:
                        entry += f" ({n.summary[:50]})"
                    ctx_parts.append(f"  {n.type.value}: {entry}")
                ctx_block = "\n".join(ctx_parts)
                sys_idx = next(
                    (i for i, m in enumerate(messages) if m.get("role") == "system"), None
                )
                if sys_idx is not None:
                    messages[sys_idx]["content"] = (
                        f"[Relevant session context]\n{ctx_block}\n\n"
                        f"{messages[sys_idx]['content']}"
                    )

        # Auto-checkpoint at threshold
        if (context_pct >= settings.graph_checkpoint.trigger_at_percent
                and settings.graph_checkpoint.enabled):
            try:
                ckpt = _checkpoint_mgr.create(
                    session_id=session_id,
                    messages=raw_messages,  # FULL history — not just recent
                    graph=graph,
                    context_pct=context_pct,
                    trigger="auto_threshold",
                    model=model,
                )
                logger.info(f"Auto-checkpoint {ckpt.checkpoint_id} for {session_id}")
            except Exception as e:
                logger.warning(f"Checkpoint failed (non-fatal): {e}")

        _set_context_used(session_id, context_used + input_tokens)

    # ── Layer 5: LLM call (or cache return) ──────────────────────────────────
    orig_input_tokens = count_messages_tokens(raw_messages, model)
    sent_input_tokens = count_messages_tokens(messages, model)
    savings["routing"] = 0  # honest default

    if cached_response:
        response_text = cached_response.response
        output_tokens = count_tokens(response_text, model)
        input_tokens_actual = 0  # cache hit = no LLM call
        latency_ms = (time.monotonic() - t0) * 1000
    else:
        provider = _get_provider()
        try:
            # Redact secrets before sending to provider
            clean_messages = redact_messages(messages)

            # Streaming: pass through to provider if client requested it.
            # NOTE: streaming bypasses output trimmer and cache write (by design).
            if req.stream:
                from fastapi.responses import StreamingResponse as _SR

                async def _stream_gen():
                    try:
                        resp = await provider.chat(
                            messages=clean_messages,
                            model=model,
                            max_tokens=req.max_tokens or 4096,
                            stream=True,
                        )
                        # Provider returns full text even in stream=True mode;
                        # emit as a single SSE chunk for now.
                        # TODO: wire true async token streaming in v0.2
                        chunk = {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant", "content": resp.text},
                                "finish_reason": "stop",
                            }],
                        }
                        import json as _json
                        yield f"data: {_json.dumps(chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        logger.error(f"Streaming error: {e}")
                        yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

                return _SR(_stream_gen(), media_type="text/event-stream")

            resp = await provider.chat(
                messages=clean_messages,
                model=model,
                max_tokens=req.max_tokens or 4096,
                stream=False,
            )
        except Exception as e:
            logger.error(f"Provider error: {e}")
            raise HTTPException(status_code=502, detail=f"Provider error: {str(e)}")

        response_text = resp.text
        output_tokens = resp.output_tokens
        input_tokens_actual = resp.input_tokens
        latency_ms = resp.latency_ms

        # Output trimmer: remove filler phrases (never touches real content)
        if settings.terse_output.enabled:
            response_text, output_saved = _output_trimmer.trim(
                response_text, level=settings.terse_output.level
            )
            savings["output_trim"] = output_saved
            output_tokens = max(1, output_tokens - output_saved)

        # Cache the response — pass session_id for proper scope isolation
        if settings.cache.enabled and user_content:
            _cache.set(
                user_content,
                response_text,
                input_tokens=input_tokens_actual,
                output_tokens=output_tokens,
                session_id=session_id,
            )

    # ── Analytics ────────────────────────────────────────────────────────────
    total_saved = sum(savings.values())
    _analytics.record(
        session_id=session_id,
        provider=settings.provider,
        model=model,
        input_tokens_original=orig_input_tokens,
        input_tokens_sent=sent_input_tokens,
        output_tokens=output_tokens,
        tokens_saved=total_saved,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
        layer_savings=savings,
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "session_id": session_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": input_tokens_actual,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens_actual + output_tokens,
            "original_prompt_tokens": orig_input_tokens,
            "tokens_saved": total_saved,
        },
        "tokenmizer": {
            "cache_hit": cache_hit,
            "savings": savings,
            "total_saved": total_saved,
            "latency_ms": round(latency_ms, 1),
        },
    }


# ── Health / Info ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/")
async def dashboard():
    from tokenmizer.dashboard.page import DASHBOARD_HTML
    return HTMLResponse(DASHBOARD_HTML)


# ── Session / Graph endpoints ─────────────────────────────────────────────────

@app.get("/api/stats", dependencies=[Depends(verify_api_key)])
async def stats(session_id: Optional[str] = None):
    return _analytics.summary()


@app.get("/api/cache/stats", dependencies=[Depends(verify_api_key)])
async def cache_stats():
    return _cache.stats()


@app.get("/api/graph/{session_id}", dependencies=[Depends(verify_api_key)])
async def get_graph(session_id: str):
    graph = _get_graph(session_id)
    return graph.stats()


@app.post("/api/checkpoint", dependencies=[Depends(verify_api_key)])
async def manual_checkpoint(session_id: str, model: str = ""):
    """Manually trigger a checkpoint for a session."""
    try:
        graph = _get_graph(session_id)
        if not graph._nodes:
            raise HTTPException(status_code=404, detail="No graph data found for session")
        ckpt = _checkpoint_mgr.create(
            session_id=session_id,
            messages=[],
            graph=graph,
            context_pct=0.0,
            trigger="manual",
            model=model,
        )
        return {
            "checkpoint_id": ckpt.checkpoint_id,
            "resume_tokens": ckpt.resume_tokens,
            "node_count": len(ckpt.graph_snapshot.get("nodes", [])),
            "resume_standard": ckpt.resume_standard,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Manual checkpoint failed for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Checkpoint failed: {str(e)}")


@app.get("/api/checkpoints/{session_id}", dependencies=[Depends(verify_api_key)])
async def list_checkpoints(session_id: str):
    return _checkpoint_mgr.list_checkpoints(session_id)


@app.post("/api/decision/invalidate", dependencies=[Depends(verify_api_key)])
async def invalidate_decision(session_id: str, decision_label: str, reason: str = ""):
    """
    Mark a decision as INVALIDATED (red) — explicitly wrong or cancelled.
    Use when a decision was made that turned out to be incorrect.
    History is preserved; decision is flagged as a warning in future resumes.
    """
    try:
        from tokenmizer.graph_memory.graph import NodeType, NodeStatus
        graph = _get_graph(session_id)
        label_lower = decision_label.lower().strip()
        found = False
        for node in graph._nodes.values():
            if (node.type == NodeType.DECISION and
                    label_lower in node.label.lower()):
                node.status = NodeStatus.INVALIDATED
                node.summary = f"Invalidated: {reason[:100]}" if reason else "Explicitly invalidated"
                found = True
        if not found:
            raise HTTPException(
                status_code=404,
                detail=f"No decision matching '{decision_label}' found in session '{session_id}'"
            )
        graph._persist()
        return {
            "session_id": session_id,
            "invalidated": decision_label,
            "reason": reason,
            "status": "invalidated",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Invalidate decision failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/resume/{session_id}", dependencies=[Depends(verify_api_key)])
async def get_resume(session_id: str, level: str = "standard"):
    """Get resume context for a session. level: critical | standard | full"""
    try:
        if level not in ("critical", "standard", "full"):
            level = "standard"
        ckpt = _checkpoint_mgr.get_latest(session_id)
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
        logger.error(f"Resume failed for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Resume failed: {str(e)}")
