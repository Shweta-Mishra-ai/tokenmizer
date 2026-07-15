"""
TokenMizer — main FastAPI application.

OpenAI-compatible proxy: POST /v1/chat/completions plus session/graph
management endpoints. See README API Reference.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from tokenmizer.analytics.engine import AnalyticsEngine
from tokenmizer.api.rate_limiter import get_rate_limiter
from tokenmizer.checkpoints.manager import CheckpointManager
from tokenmizer.compression.engine import CompressionPipeline
from tokenmizer.compression.output_trimmer import OutputTrimmer
from tokenmizer.compression.window import SmartMessageWindow, needs_windowing
from tokenmizer.config.settings import get_settings
from tokenmizer.core.tokenizer import count_messages_tokens, count_tokens
from tokenmizer.filters.file_intelligence import FileIntelligence
from tokenmizer.graph_memory.graph import GraphMemory
from tokenmizer.providers.providers import build_provider
from tokenmizer.security.auth import verify_api_key
from tokenmizer.security.middleware import injection_guard
from tokenmizer.security.redaction import redact_messages
from tokenmizer.semantic_cache.cache import SemanticCache
from tokenmizer.state.backend import get_state_backend

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
# Max 1000 concurrent sessions; LRU eviction removes oldest UNHELD lock.
_SESSION_LOCK_MAX = 1000
_session_locks: "OrderedDict[str, asyncio.Lock]" = OrderedDict()


def _get_session_lock(session_id: str) -> asyncio.Lock:
    """
    Get or create a per-session async lock (LRU-bounded).

    Eviction safety: only evicts UNHELD locks (lock.locked() == False).
    If all locks happen to be held when over the cap (extremely unlikely
    at 1000 concurrent sessions), we skip eviction this call rather than
    risk a held lock being dropped — which would let a new request bypass
    an in-flight request's mutual exclusion for the same session.
    """
    if session_id in _session_locks:
        _session_locks.move_to_end(session_id)
        return _session_locks[session_id]

    lock = asyncio.Lock()
    _session_locks[session_id] = lock

    if len(_session_locks) > _SESSION_LOCK_MAX:
        # Find oldest unheld lock to evict (iterate from front)
        for old_id in list(_session_locks.keys()):
            if old_id == session_id:
                continue
            if not _session_locks[old_id].locked():
                del _session_locks[old_id]
                break
            if len(_session_locks) <= _SESSION_LOCK_MAX:
                break

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

    from tokenmizer.providers.providers import AnthropicProvider, OpenAIProvider

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

# LRU-bounded cache of GraphMemory objects (evicts least-recently-used).
# Graph data is persisted to SQLite, so eviction just frees memory —
# the graph reloads from disk on next access. Cap chosen for typical
# self-hosted deployments (one process, many sessions over time).
_GRAPH_CACHE_MAX = 200
_graph_cache: "OrderedDict[str, GraphMemory]" = OrderedDict()
_graph_cache_lock = asyncio.Lock()  # guards dict creation — prevents TOCTOU race


def _graph_cache_touch(session_id: str) -> None:
    """Move session to end (most-recently-used) and evict oldest if over cap."""
    _graph_cache.move_to_end(session_id)
    while len(_graph_cache) > _GRAPH_CACHE_MAX:
        evicted_id, evicted_graph = _graph_cache.popitem(last=False)
        # Ensure pending writes are flushed before dropping from memory.
        #
        # FIXED: previously a failed flush here was caught, logged at
        # `error`, and then the graph was dropped from memory anyway —
        # meaning any nodes added since the last successful `_persist()`
        # call were gone permanently, with zero visibility beyond a log
        # line. This is silent, permanent data loss in a tool whose whole
        # pitch is "never lose context." We now retry once (covers
        # transient SQLite WAL lock contention) and record the failure to
        # analytics so it's queryable via /api/stats instead of invisible.
        persisted = False
        for attempt in range(2):
            try:
                evicted_graph._persist()
                persisted = True
                break
            except Exception as e:
                if attempt == 0:
                    logger.warning(
                        f"Persist attempt 1 failed for evicted graph {evicted_id}, retrying: {e}"
                    )
                else:
                    logger.error(
                        f"Graph {evicted_id} evicted from cache WITHOUT persisting — "
                        f"nodes added since last successful save are LOST: {e}"
                    )
        if not persisted:
            _analytics.record_silent_failure("graph_eviction")


async def _get_graph_async(session_id: str) -> GraphMemory:
    """
    Race-safe, LRU-bounded graph accessor for async handlers.
    Double-checked locking: avoids creating two GraphMemory objects
    for the same session when concurrent requests both see a cache miss.
    """
    if session_id in _graph_cache:
        _graph_cache_touch(session_id)
        return _graph_cache[session_id]
    async with _graph_cache_lock:
        if session_id not in _graph_cache:  # re-check after lock
            _graph_cache[session_id] = GraphMemory(
                session_id,
                storage_dir=settings.graph_checkpoint.storage_dir,
            )
        _graph_cache_touch(session_id)
        return _graph_cache[session_id]


def _get_context_used(session_id: str) -> int:
    return _state.get(f"ctx:{session_id}") or 0


def _set_context_used(session_id: str, tokens: int) -> None:
    # FIXED: state backend `set()` now returns bool (see state/backend.py).
    # A dropped write here under-counts context usage, which can silently
    # cause the auto-checkpoint trigger_at_percent threshold to be missed —
    # the proxy thinks the session has used less context than it actually
    # has. Recording the failure makes this visible via /api/stats instead
    # of manifesting only as "why didn't my checkpoint fire."
    ok = _state.set(f"ctx:{session_id}", tokens, ttl=86400)
    if not ok:
        logger.error(f"Failed to persist context usage for session {session_id}")
        _analytics.record_silent_failure("state_backend_set")


# ── Context window sizes ──────────────────────────────────────────────────────

# Newest Claude models (fable-5, opus-4-8, sonnet-5, haiku-4-5) all match the
# "claude" prefix entry. Add a specific entry ONLY if a model's window differs.
_CONTEXT_WINDOWS = {
    "claude-fable-5": 200_000, "claude-opus-4-8": 200_000,
    "claude-sonnet": 200_000, "claude-opus": 200_000, "claude-haiku": 200_000,
    "claude": 200_000,
    "gpt-4o": 128_000, "gpt-4": 128_000, "gpt-3.5": 16_000,
    "gemini": 1_000_000, "deepseek": 64_000,
}


def _context_window(model: str) -> int:
    # Longest key first — so "claude-fable-5" wins over the "claude" catch-all
    # if their values ever diverge. (Previously dict order decided; the broad
    # "claude" key shadowed every specific entry.)
    m = model.lower()
    for k in sorted(_CONTEXT_WINDOWS, key=len, reverse=True):
        if k in m:
            return _CONTEXT_WINDOWS[k]
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
    version="0.4.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,  # defaults: localhost:3000, localhost:8000
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Session-ID", "X-API-Key"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """OpenAI-style message. `content` accepts a plain string OR a list of
    content blocks (multimodal format). Blocks are normalized to text —
    TokenMizer is a text proxy; non-text blocks (images) are dropped with
    their text parts preserved."""
    role: str
    content: str | list | None = ""

    def text(self) -> str:
        from tokenmizer.graph_memory.helpers import _content_to_text
        return _content_to_text(self.content)


class ChatRequest(BaseModel):
    """OpenAI-compatible request. Sampling params (temperature, top_p, stop)
    are forwarded to the provider. Unknown fields are accepted and ignored
    (extra='allow') so standard OpenAI clients never get a 422 — but only
    the fields below influence the call."""
    model_config = {"extra": "allow"}

    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    session_id: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[str | list[str]] = None


def _sampling_kwargs(req: "ChatRequest") -> dict:
    """Sampling params to forward to the provider (only ones explicitly set)."""
    kw: dict = {}
    if req.temperature is not None:
        kw["temperature"] = req.temperature
    if req.top_p is not None:
        kw["top_p"] = req.top_p
    if req.stop is not None:
        kw["stop"] = req.stop
    return kw


# ── chat_completions helpers ──────────────────────────────────────────────────


async def _check_rate_limit(request: Request) -> None:
    """Raise 429 if client is rate-limited."""
    client_ip = request.client.host if request.client else "unknown"
    client_id = request.headers.get("Authorization", client_ip)
    allowed, retry_after = await _rate_limiter.check(client_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after:.1f}s",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )


def _apply_compression_layers(
    messages: list[dict],
    settings,
    savings: dict,
) -> list[dict]:
    """
    Layer 0-2: file intelligence, compression, terse injection.
    Returns compressed messages and populates savings dict.
    """
    user_query = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
    )
    # Layer 0: File intelligence
    messages, file_saved = _file_intelligence.process_message_files(
        messages, token_budget_per_file=600, query=user_query
    )
    savings["file_extraction"] = file_saved

    # Layer 1: Prompt compression
    if settings.compression.enabled:
        compressed, saved = _compression.compress_messages(messages, protect_recent=3)
        messages = compressed
        savings["compression"] = saved

    # Layer 2: Terse output injection
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

    return messages


async def _update_graph(
    session_id: str,
    graph,
    raw_messages: list[dict],
    messages: list[dict],
    model: str,
    savings: dict,
    user_query: str,
) -> tuple[list[dict], dict]:
    """
    Layer 4: Graph extraction, smart windowing, context injection, checkpoint.
    Mutates messages (adds graph context).
    Returns (updated_messages, checkpoint_status) — checkpoint_status surfaces
    auto-checkpoint success/failure to the caller instead of only logging it.
    """
    context_used   = _get_context_used(session_id)
    context_window = _context_window(model)
    input_tokens   = count_messages_tokens(messages, model)
    context_pct    = (context_used + input_tokens) / context_window

    # Extraction: heuristic sync now, LLM async in background
    if settings.graph_checkpoint.use_llm_extraction:
        cheap = _get_cheap_provider()
        if cheap is not None:
            recent = raw_messages[-4:] if len(raw_messages) >= 4 else raw_messages
            new_msgs = [m for m in recent
                        if graph._msg_hash(m) not in graph._processed_hashes]
            if new_msgs:
                graph.extract_from_messages(raw_messages, incremental=True)
                _lock_ref = _get_session_lock(session_id)

                async def _background_extract(
                    _g=graph, _msgs=new_msgs, _all=raw_messages,
                    _cheap=cheap, _lock=_lock_ref, _sid=session_id,
                ):
                    async with _lock:
                        try:
                            from tokenmizer.graph_memory.hybrid_extractor import HybridExtractor

                            async def _pfn(messages, system="", max_tokens=600):
                                r = await _cheap.chat(
                                    messages=messages, system=system, max_tokens=max_tokens
                                )
                                return {"text": r.text}

                            # provider_fn goes to extract(), not __init__ —
                            # omitting it silently skips the LLM pass
                            # (regression-tested in test_hybrid_extractor).
                            ext = HybridExtractor()
                            extracted = await ext.extract(_msgs, provider_fn=_pfn)
                            _g.extract_from_messages(_all, incremental=False,
                                                     extracted_data=extracted)
                            logger.debug(f"HybridExtractor complete for {_sid}")
                        except Exception as e:
                            # FIXED: previously logged at `debug` (off by
                            # default in production) — meaning the entire
                            # LLM-powered extraction feature could fail on
                            # every single call (e.g. invalid/expired cheap-
                            # provider API key, provider outage, quota
                            # exhausted) and run silently for the whole
                            # session with zero visibility anywhere. The
                            # graph would just quietly stop gaining new
                            # nodes from this path and nobody would know
                            # why. Bumped to `warning` (visible by default)
                            # and tracked via analytics so persistent
                            # failures are queryable via /api/stats instead
                            # of only discoverable by reading debug logs.
                            logger.warning(
                                f"Background LLM extraction failed for session "
                                f"{_sid} (falling back to heuristic-only on next "
                                f"calls, no data lost — just less accurate "
                                f"extraction this turn): {e}"
                            )
                            _analytics.record_silent_failure("llm_extraction")

                asyncio.create_task(_background_extract())
            else:
                graph.extract_from_messages(raw_messages, incremental=True)
        else:
            graph.extract_from_messages(raw_messages, incremental=True)
    else:
        graph.extract_from_messages(raw_messages, incremental=True)

    # Smart windowing
    if needs_windowing(messages, settings.memory.max_tokens_before_summary, model):
        messages, window_saved = _smart_window.apply(messages, graph, model)
        savings["windowing"] = window_saved
    else:
        savings["windowing"] = 0

    # Context injection — only when graph has enough signal
    if len(graph._nodes) >= 3 and len(user_query.split()) >= 4:
        relevant = graph.query(user_query, top_k=8)
        if relevant:
            ctx_parts = [
                f"  {n.type.value}: {n.label}"
                + (f" ({n.summary[:50]})" if n.summary else "")
                for n in relevant[:6]
            ]
            ctx_block = "\n".join(ctx_parts)
            sys_idx = next(
                (i for i, m in enumerate(messages) if m.get("role") == "system"), None
            )
            if sys_idx is not None:
                messages[sys_idx]["content"] = (
                    f"[Relevant session context]\n{ctx_block}\n\n"
                    f"{messages[sys_idx]['content']}"
                )

    # Auto-checkpoint
    #
    # FIXED: previously a failed auto-checkpoint was caught, logged at
    # `warning`, and otherwise invisible — the chat response returned
    # normally with no indication that the safety net didn't fire. For a
    # tool whose entire pitch is "never lose context across sessions,"
    # silently failing the auto-checkpoint and telling the user nothing is
    # the single worst failure mode this codebase had. The chat request
    # still should NOT fail just because the checkpoint failed (the user
    # came here for an answer, not a checkpoint), but the failure must be
    # visible somewhere the caller can actually see it.
    #
    # Fix: retry once (covers transient SQLite lock contention under
    # concurrent requests — see WAL mode notes in checkpoints/manager.py),
    # log at `error` if it still fails, and record the failure in `savings`
    # so it flows into the `tokenmizer.checkpoint` response field below —
    # a client that cares can check `checkpoint_failed` instead of having
    # to grep server logs to discover their context wasn't saved.
    checkpoint_status = {"attempted": False, "succeeded": False, "checkpoint_id": None}
    if (context_pct >= settings.graph_checkpoint.trigger_at_percent
            and settings.graph_checkpoint.enabled):
        checkpoint_status["attempted"] = True
        last_error: Optional[Exception] = None
        for attempt in range(2):  # one retry for transient SQLite lock contention
            try:
                ckpt = _checkpoint_mgr.create(
                    session_id=session_id,
                    messages=raw_messages,
                    graph=graph,
                    context_pct=context_pct,
                    trigger="auto_threshold",
                    model=model,
                )
                logger.info(f"Auto-checkpoint {ckpt.checkpoint_id} for {session_id}")
                checkpoint_status["succeeded"] = True
                checkpoint_status["checkpoint_id"] = ckpt.checkpoint_id
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt == 0:
                    logger.warning(
                        f"Auto-checkpoint attempt 1 failed for {session_id}, retrying once: {e}"
                    )
                await asyncio.sleep(0.1)
        if last_error is not None:
            logger.error(
                f"Auto-checkpoint FAILED for {session_id} after retry — "
                f"context was NOT saved at {context_pct:.0%} usage: {last_error}"
            )
            checkpoint_status["error"] = str(last_error)
            _analytics.record_silent_failure("checkpoint")

    _set_context_used(session_id, context_used + input_tokens)
    return messages, checkpoint_status


async def _call_provider(
    req,
    messages: list[dict],
    model: str,
    user_content: str,
    session_id: str,
    savings: dict,
) -> tuple[str, int, int, float]:
    """
    Layer 3 + 5: Cache lookup → LLM call → output trim → cache write.
    Returns (response_text, input_tokens, output_tokens, latency_ms).
    """
    # Cache lookup
    if settings.cache.enabled and user_content:
        cached = _cache.get(user_content, session_id=session_id)
        if cached:
            savings["cache"] = count_tokens(user_content, model)
            output_tokens = count_tokens(cached.response, model)
            return cached.response, 0, output_tokens, 0.0

    # LLM call
    # NOTE: `messages` is already redacted — redaction now happens once at
    # ingestion in chat_completions() so every downstream consumer (this call,
    # background graph extraction, checkpoint storage) sees the same safe
    # copy. We do NOT re-redact here to avoid masking a regression upstream:
    # if redaction is ever accidentally removed at ingestion, this call site
    # should not silently paper over it.
    provider = _get_provider()
    try:
        resp  = await provider.chat(
            messages=messages, model=model,
            max_tokens=req.max_tokens or 4096, stream=False,
            **_sampling_kwargs(req),
        )
    except Exception as e:
        logger.error(f"Provider error: {e}")
        raise HTTPException(status_code=502, detail=f"Provider error: {str(e)}")

    response_text  = resp.text
    output_tokens  = resp.output_tokens
    input_tokens   = resp.input_tokens
    latency_ms     = resp.latency_ms

    # Output trim
    if settings.terse_output.enabled:
        response_text, output_saved = _output_trimmer.trim(
            response_text, level=settings.terse_output.level
        )
        savings["output_trim"] = output_saved
        output_tokens = max(1, output_tokens - output_saved)

    # Cache write
    if settings.cache.enabled and user_content:
        _cache.set(user_content, response_text,
                   input_tokens=input_tokens, output_tokens=output_tokens,
                   session_id=session_id)

    return response_text, input_tokens, output_tokens, latency_ms


def _stream_response(req, messages, model, user_content, session_id,
                     savings, orig_input_tokens) -> StreamingResponse:
    """True SSE passthrough (OpenAI chat.completion.chunk format).

    Cache hits stream as a single chunk. After the stream closes, analytics
    and the semantic-cache write run on the accumulated text — same
    bookkeeping as the non-stream path, minus output trimming.
    """
    import json as _json

    from tokenmizer.providers.providers import BaseProvider, ProviderError

    provider = _get_provider()
    if getattr(type(provider), "chat_stream", None) is BaseProvider.chat_stream:
        raise HTTPException(
            status_code=501,
            detail=(f"Streaming passthrough is not implemented for provider "
                    f"'{settings.provider}' yet (supported: anthropic, openai, "
                    f"deepseek, mistral, openrouter, grok, ollama). "
                    f"Set stream=false for this provider."),
        )

    resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def _chunk(delta: dict, finish: str | None = None) -> str:
        return "data: " + _json.dumps({
            "id": resp_id, "object": "chat.completion.chunk",
            "created": created, "model": model, "session_id": session_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }) + "\n\n"

    async def _gen():
        full_text = ""
        t0 = time.monotonic()
        cache_hit = False
        yield _chunk({"role": "assistant"})
        try:
            cached = (
                _cache.get(user_content, session_id=session_id)
                if settings.cache.enabled and user_content
                else None
            )

            cache_hit = cached is not None

            if cached:
                full_text = cached.response
                yield _chunk({"content": full_text})
            else:
                async for piece in provider.chat_stream(
                    messages=messages,
                    model=model,
                    max_tokens=req.max_tokens or 4096,
                    **_sampling_kwargs(req),
                ):
                    full_text += piece
                    yield _chunk({"content": piece})

        except ProviderError as e:
            # Mid-stream failure: SSE can't change the status code anymore —
            #emit an explicit error event instead of silently truncating.
            yield "data: " + _json.dumps(
                {"error": {"message": str(e), "type": "provider_error"}}
            ) + "\n\n"
        yield _chunk({}, finish="stop")
        yield "data: [DONE]\n\n"

        # Post-stream bookkeeping
        latency_ms = (time.monotonic() - t0) * 1000
        output_tokens = count_tokens(full_text, model)
        input_tokens = count_messages_tokens(messages, model)
        if settings.cache.enabled and user_content and full_text:
            _cache.set(user_content, full_text, input_tokens=input_tokens,
                       output_tokens=output_tokens, session_id=session_id)
        _analytics.record(
            session_id=session_id, provider=settings.provider, model=model,
            input_tokens_original=orig_input_tokens,
            input_tokens_sent=input_tokens, output_tokens=output_tokens,
            tokens_saved=sum(savings.values()), latency_ms=latency_ms,
            cache_hit=cache_hit, layer_savings=savings,
        )

    return StreamingResponse(_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key), Depends(injection_guard)])
async def chat_completions(req: ChatRequest, request: Request):
    """
    Main proxy endpoint — orchestrates all 6 layers.

    Split into helpers to keep this orchestrator readable:
      _check_rate_limit()       — 429 if over limit
      _apply_compression_layers() — file intelligence, compress, terse inject
      _update_graph()           — graph extraction, windowing, context inject
      _call_provider()          — cache → LLM → output trim → cache write
    """
    session_id = req.session_id or str(uuid.uuid4())
    model      = req.model or settings.default_model
    savings: dict[str, int] = {}

    await _check_rate_limit(request)

    # SECURITY: redact secrets/PII at the earliest possible point, before
    # ANY downstream consumer sees the content. This includes:
    #   - the main chat provider call (_call_provider)
    #   - the background graph-extraction LLM call (_update_graph → HybridExtractor),
    #     which talks to a *separate*, often cheaper third-party model
    #     (haiku/gpt-4o-mini/deepseek) — previously this saw RAW unredacted
    #     content because only _call_provider redacted its own copy.
    #   - checkpoint storage (SQLite) and the graph DB itself
    # Redacting once here means every downstream path is safe by construction
    # instead of relying on each call site to remember to redact.
    raw_messages = [{"role": m.role, "content": m.text()} for m in req.messages]
    raw_messages = redact_messages(raw_messages)
    messages     = raw_messages[:]
    user_query   = next(
        (m["content"] for m in reversed(raw_messages) if m.get("role") == "user"), ""
    )
    user_content = user_query

    # Layer 0-2: file intelligence, compression, terse injection
    messages = _apply_compression_layers(messages, settings, savings)

    # Layer 3+5: cache + LLM + output trim (done before graph for latency)
    # Graph runs in parallel-ish: heuristic extract is sync and fast,
    # LLM extract fires async after provider returns.
    orig_input_tokens = count_messages_tokens(raw_messages, model)
    sent_input_tokens = count_messages_tokens(messages, model)
    savings["routing"] = 0

    # Layer 4: graph update + context injection (mutates messages)
    checkpoint_status: dict = {"attempted": False, "succeeded": False, "checkpoint_id": None}
    if settings.graph_checkpoint.enabled:
        graph    = await _get_graph_async(session_id)
        messages, checkpoint_status = await _update_graph(
            session_id, graph, raw_messages, messages, model, savings, user_query
        )

    # Streaming: true SSE passthrough (v0.3). Output-trimming is skipped in
    # stream mode (can't trim tokens that already left the building) — all
    # input-side layers (file intel, compression, graph context) still apply.
    if req.stream:
        return _stream_response(req, messages, model, user_content,
                                session_id, savings, orig_input_tokens)

    # Layer 5: call provider (or return cache hit)
    response_text, input_tokens_actual, output_tokens, latency_ms = await _call_provider(
        req, messages, model, user_content, session_id, savings
    )
    cache_hit = input_tokens_actual == 0 and response_text != ""

    # Analytics
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
        "id":      f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   model,
        "session_id": session_id,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":          input_tokens_actual,
            "completion_tokens":      output_tokens,
            "total_tokens":           input_tokens_actual + output_tokens,
            "original_prompt_tokens": orig_input_tokens,
            "tokens_saved":           total_saved,
        },
        "tokenmizer": {
            "cache_hit":   cache_hit,
            "savings":     savings,
            "total_saved": total_saved,
            "latency_ms":  round(latency_ms, 1),
            # FIXED: previously a failed auto-checkpoint was invisible to
            # the caller — only a log line nobody watches. Now surfaced
            # here so a client can detect "my context wasn't saved" instead
            # of finding out only when resume returns nothing.
            "checkpoint": checkpoint_status,
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
    stats = _cache.stats()
    # Include preference context for completeness (was previously unused)
    stats["preference_context"] = _cache._preference_store.to_system_context()
    return stats


@app.get("/api/graph/{session_id}/history", dependencies=[Depends(verify_api_key)])
async def get_graph_history(session_id: str, at_time: float = 0.0, top_k: int = 12):
    """
    Query graph state at a specific Unix timestamp.
    at_time=0.0 (default) returns current state (equivalent to /viz).
    at_time=<unix_ts> returns which nodes were active at that point in time.

    Useful for: debugging decision changes, audit trail, "what did we decide
    at 2pm?" queries.
    """
    graph = await _get_graph_async(session_id)
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


@app.get("/api/graph/{session_id}", dependencies=[Depends(verify_api_key)])
async def get_graph(session_id: str):
    graph = await _get_graph_async(session_id)
    return graph.stats()


@app.get("/api/graph/{session_id}/viz", dependencies=[Depends(verify_api_key)])
async def get_graph_viz(session_id: str):
    """
    Return full graph as D3-compatible JSON for visualization.
    {nodes: [...], edges: [...], meta: {...}}
    Used by the dashboard Graph tab and any external viz tool.
    """
    graph = await _get_graph_async(session_id)
    return graph.to_vis_json()


@app.get("/api/graph/{session_id}/html", dependencies=[Depends(verify_api_key)])
async def get_graph_html(session_id: str):
    """Shareable standalone interactive graph — open in a browser, drag/zoom,
    screenshot, share. Self-contained dark-theme D3 force layout."""
    from tokenmizer.graph_memory.visualization import to_share_html
    graph = await _get_graph_async(session_id)
    return HTMLResponse(to_share_html(graph))


@app.get("/api/ontology", dependencies=[Depends(verify_api_key)])
async def get_ontology():
    """The TokenMizer graph ontology: node/edge types with semantics and
    the status state machine. Machine-readable — what the graph CAN contain
    and which lifecycle transitions are legal."""
    from tokenmizer.graph_memory.ontology import ontology_dict
    return ontology_dict()


@app.get("/api/graph/{session_id}/why", dependencies=[Depends(verify_api_key)])
async def get_graph_why(session_id: str, q: str):
    """Reasoning: trace the causal chain behind a decision. Matches decision
    nodes containing `q`, walks the supersession chain in both directions,
    and returns the old→new trail with trigger/reason/evidence per hop,
    plus the currently active choice."""
    from tokenmizer.graph_memory.reasoning import why
    graph = await _get_graph_async(session_id)
    return why(graph, q)


@app.get("/api/graph/{session_id}/reasoning", dependencies=[Depends(verify_api_key)])
async def get_graph_reasoning(session_id: str):
    """Full reasoning view over session memory: active decisions, recent
    changes, decision history grouped by topic, and an ontology-based
    consistency audit (contradictions, missing/dangling transitions)."""
    from tokenmizer.graph_memory.reasoning import summarize_reasoning
    graph = await _get_graph_async(session_id)
    return summarize_reasoning(graph)


@app.get("/api/graph/{session_id}/obsidian", dependencies=[Depends(verify_api_key)])
async def get_graph_obsidian(session_id: str):
    """
    Download graph as Obsidian Canvas (.canvas) file.
    Save as <any-name>.canvas inside your Obsidian vault and open directly.
    """
    import json as _json

    from fastapi.responses import Response as _Resp
    graph = await _get_graph_async(session_id)
    canvas = graph.to_obsidian_canvas()
    filename = f"tokenmizer-{session_id[:12]}.canvas"
    return _Resp(
        content=_json.dumps(canvas, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/api/graph/{session_id}/transitions", dependencies=[Depends(verify_api_key)])
async def get_transitions(session_id: str):
    """Full decision transition history — trigger, reason, evidence, confidence_delta."""
    graph = await _get_graph_async(session_id)
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



@app.post("/api/checkpoint", dependencies=[Depends(verify_api_key)])
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
        graph = await _get_graph_async(session_id)
        ckpt = _checkpoint_mgr.create(
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
        logger.error(f"Manual checkpoint failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        from tokenmizer.graph_memory.graph import NodeStatus, NodeType
        graph = await _get_graph_async(session_id)
        label_lower = decision_label.lower().strip()
        # Only ACTIVE (COMPLETED) decisions are eligible: matching across
        # all statuses would let a short label overwrite SUPERSEDED history
        # nodes with INVALIDATED, destroying their supersession record. The
        # response lists every affected node so multi-matches are visible.
        invalidated: list[dict] = []
        for node_id, node in graph._nodes.items():
            if (node.type == NodeType.DECISION and
                    node.status == NodeStatus.COMPLETED and
                    label_lower in node.label.lower()):
                node.status = NodeStatus.INVALIDATED
                node.summary = (
                    f"Invalidated: {reason[:100]}" if reason else "Explicitly invalidated"
                )
                invalidated.append({"node_id": node_id, "label": node.label})
        if not invalidated:
            raise HTTPException(
                status_code=404,
                detail=(f"No ACTIVE decision matching '{decision_label}' found in "
                        f"session '{session_id}' (superseded/archived decisions "
                        f"are not invalidatable — they are already inactive)")
            )
        graph._persist(force=True)  # direct node mutation above bypasses add_node's
                                      # dirty-tracking — force=True is required here or
                                      # this write is silently skipped (caught in a final
                                      # accuracy pass; same class of bug the eviction path
                                      # and prune() were already protected against)
        return {
            "session_id": session_id,
            "invalidated": decision_label,
            "affected_nodes": invalidated,
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
