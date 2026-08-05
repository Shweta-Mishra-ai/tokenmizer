"""
TokenMizer — main FastAPI application.

OpenAI-compatible proxy: POST /v1/chat/completions, plus app setup,
singletons, and the shared helpers (_get_graph_async, _check_rate_limit,
_update_graph, etc.) that both this file and routes_graph.py depend on.
See README API Reference.

Session/graph inspection, checkpoint, and decision-management endpoints
(~15 routes) live in routes_graph.py — split out to keep this file
focused on the core proxy path. That module is imported at the bottom
of this file and references the singletons/helpers defined here via
`app_module.<name>` rather than importing them by value, so tests that
monkeypatch this module's attributes (e.g. `app_module._analytics`,
`app_module._graph_cache`) keep working unchanged regardless of which
file actually handles a given request.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager, contextmanager
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
from tokenmizer.security.ownership import (
    DEV_PRINCIPAL,
    OwnershipStore,
    OwnershipUnavailable,
    SessionAccessDenied,
)
from tokenmizer.security.redaction import redact_messages
from tokenmizer.semantic_cache.cache import SemanticCache

logger = logging.getLogger(__name__)

settings = get_settings()


def _warn_if_multi_worker_risk() -> None:
    """
    Best-effort startup check for a risk that cannot be fixed with an
    in-process lock: running multiple OS worker processes (e.g. `uvicorn
    --workers 4`) against the same storage_dir. Each worker gets its own
    in-process `_graph_cache` and its own GraphMemory instances for the
    same session_id — genuinely separate memory across processes — and
    the current full-blob SQLite persist (see issue #27) means whichever
    process saves a session last silently overwrites whatever an earlier
    worker wrote. No amount of asyncio.Lock helps here; locks don't cross
    process boundaries.

    This can't be detected with certainty from inside a single process
    (there's no universal "how many workers am I one of" signal), so this
    checks a few common launcher env vars as a heuristic. A false negative
    (multi-worker deployment this doesn't catch) is expected for unusual
    launch setups — the goal is to catch the common case loudly rather
    than stay silent about a real risk, not to guarantee detection.
    """
    import os
    for var in ("WEB_CONCURRENCY", "UVICORN_WORKERS", "GUNICORN_WORKERS"):
        val = os.environ.get(var, "").strip()
        if val.isdigit() and int(val) > 1:
            logger.warning(
                f"{var}={val} suggests multiple worker processes. "
                "TokenMizer's SQLite-backed graph/checkpoint storage is "
                "NOT safe for concurrent multi-process writers to the "
                "same session — the last process to persist a session "
                "silently overwrites what an earlier one wrote (tracked "
                "as issue #27, which will move to per-row persistence). "
                "Until that lands: run a single worker per storage_dir, or "
                "route each session_id to a fixed worker (e.g. consistent "
                "hashing at your load balancer)."
            )
            return


_warn_if_multi_worker_risk()

# ── Singletons ────────────────────────────────────────────────────────────────
_provider = None
_compression = CompressionPipeline(
    ratio=settings.compression.ratio,
    enable_ml=(settings.compression.engine == "llmlingua2"),
    min_tokens_to_compress=settings.compression.min_tokens_to_compress,
)
_cache = SemanticCache(
    threshold=settings.cache.similarity_threshold,
    ttl_seconds=settings.cache.ttl_seconds,
    max_size=settings.cache.max_size,
    share_scope=settings.cache.share_scope,
)
_checkpoint_mgr = CheckpointManager(
    storage_dir=settings.graph_checkpoint.storage_dir,
    max_resume_tokens=settings.graph_checkpoint.max_resume_tokens,
)
_ownership = OwnershipStore(storage_dir=settings.graph_checkpoint.storage_dir)
_analytics = AnalyticsEngine()
_output_trimmer = OutputTrimmer()
_rate_limiter = get_rate_limiter(rate=60, per_seconds=60, burst=10)

# Bounded LRU for session locks — prevents memory leak on long-running servers.
# Max 1000 concurrent sessions; LRU eviction removes oldest UNHELD lock.
_SESSION_LOCK_MAX = 1000
_session_locks: "OrderedDict[str, asyncio.Lock]" = OrderedDict()

# Strong references to in-flight background tasks (e.g. the LLM
# extraction pass scheduled by _update_graph). asyncio.create_task()'s own
# docs warn that a task with no reference held anywhere is eligible for
# garbage collection before it completes, silently dropping whatever it
# was doing — for the background extraction task, that would mean the
# graph quietly stops gaining nodes from this path with no error at all.
# Each task removes itself via the done-callback, so this set never grows
# unbounded.
_background_tasks: set[asyncio.Task] = set()


def _track_background_task(coro) -> asyncio.Task:
    """asyncio.create_task() + strong-reference retention in one call —
    use this instead of a bare create_task() for any fire-and-forget task
    that must not be garbage-collected before it finishes."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


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
        # Find oldest unheld lock to evict (iterate from front). Note: the
        # trailing "already back under cap" check that used to sit here
        # was unreachable dead code — the only way `len()` changes inside
        # this loop is the `del` immediately above, which is followed by
        # an unconditional `break` in the same branch, so that second
        # check could never be reached with a smaller length than when
        # the loop started.
        for old_id in list(_session_locks.keys()):
            if old_id == session_id:
                continue
            if not _session_locks[old_id].locked():
                del _session_locks[old_id]
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

    # `graph_checkpoint.extraction_model` is documented as "leave empty =
    # auto-pick cheapest model for your provider", implying a non-empty
    # value is honoured. Nothing read it — the models below were
    # hardcoded, so anyone pinning a specific extraction model was
    # silently ignored. An explicit value now wins; empty keeps the
    # documented auto-pick.
    override = (settings.graph_checkpoint.extraction_model or "").strip()

    if provider in ("anthropic", "claude") and key:
        _cheap_provider = AnthropicProvider(key, model=override or "claude-haiku-4-5")
    elif provider in ("openai", "gpt") and key:
        _cheap_provider = OpenAIProvider(key, model=override or "gpt-4o-mini")
    elif provider == "deepseek" and key:
        from tokenmizer.providers.providers import DeepSeekProvider
        _cheap_provider = DeepSeekProvider(key, model=override or "deepseek-chat")
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
#
# Concurrency note (corrected — the previous comment here overstated its
# own mechanism): concurrent asyncio coroutines mutating the SAME
# GraphMemory object do not actually corrupt its dicts, but not because of
# `_get_session_lock()` — that lock is only acquired by the background
# LLM-extraction task, not by the request-handling path. The real reason
# it's safe is that GraphMemory's mutation methods (add_node, add_edge,
# extract_from_messages, etc.) contain no internal `await`, and CPython's
# cooperative scheduler only switches between coroutines at await points —
# so two coroutines' calls into the same object can never interleave
# mid-method. This was verified empirically (not assumed) during the
# audit: concurrent add_node() calls across 50 coroutines never dropped a
# node. What IS a real risk — and what `_graph_cache_touch()` below
# actually guards against — is EVICTING a GraphMemory instance (and force-
# persisting its current state) while a request or the background task
# still holds that session's lock, i.e. is actively using it. If that
# instance is evicted and the session is re-fetched, a fresh GraphMemory
# reloads from disk; the original evicted instance's in-flight mutation
# can then persist AFTER the new instance already started writing,
# silently clobbering it. So eviction must skip any session whose lock is
# currently held — mirroring the same rule `_session_locks` already
# applies to itself in `_get_session_lock()` above.

# LRU-bounded cache of GraphMemory objects (evicts least-recently-used).
# Graph data is persisted to SQLite, so eviction just frees memory —
# the graph reloads from disk on next access. Cap chosen for typical
# self-hosted deployments (one process, many sessions over time).
_GRAPH_CACHE_MAX = 200
_graph_cache: "OrderedDict[str, GraphMemory]" = OrderedDict()
_graph_cache_lock = asyncio.Lock()  # guards dict creation — prevents TOCTOU race


# Per-session in-flight request counter.
#
# The previous "is this session in use?" test read _session_locks — but
# as the note above admits, that lock is ONLY taken by the background
# extraction task, never by the request path. So the guard it powered
# was inert for ordinary requests: a request holding a GraphMemory
# reference could have that instance evicted and force-persisted
# underneath it, then add nodes to the now-detached object while a fresh
# instance reloaded from disk and wrote over them. Requests now mark
# themselves via _session_in_use(), so the check reflects reality.
_session_inflight: dict[str, int] = {}


@contextmanager
def _session_in_use(session_id: str):
    """Mark a session as actively used by a request for the duration of
    the block, so graph-cache eviction leaves its GraphMemory alone."""
    _session_inflight[session_id] = _session_inflight.get(session_id, 0) + 1
    try:
        yield
    finally:
        remaining = _session_inflight.get(session_id, 1) - 1
        if remaining <= 0:
            _session_inflight.pop(session_id, None)
        else:
            _session_inflight[session_id] = remaining


def _find_evictable_graph_id() -> Optional[str]:
    """First (most-LRU) session_id in _graph_cache that no request and no
    background task is currently using, or None if every remaining entry
    is in-flight. See the concurrency note above for why this exists."""
    for sid in _graph_cache:  # OrderedDict: iteration order == LRU order
        if _session_inflight.get(sid):
            continue
        lock = _session_locks.get(sid)
        if lock is None or not lock.locked():
            return sid
    return None


def _graph_cache_touch(session_id: str) -> None:
    """Move session to end (most-recently-used) and evict oldest unheld
    entries if over cap. If every entry over the cap is currently
    in-flight (its session lock is held), eviction is skipped for this
    call rather than risk evicting/force-persisting a session mid-use —
    the cache will simply run one entry over cap until something frees up,
    which is a far smaller cost than silent cross-instance data loss."""
    _graph_cache.move_to_end(session_id)
    while len(_graph_cache) > _GRAPH_CACHE_MAX:
        evicted_id = _find_evictable_graph_id()
        if evicted_id is None:
            logger.debug(
                "Graph cache over cap but every entry is in-flight "
                "(session lock held) — skipping eviction this call"
            )
            break
        evicted_graph = _graph_cache.pop(evicted_id)
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
        #
        # FIXED (TM-12): _persist() used to catch its own exceptions and
        # return None unconditionally, so this try/except never actually
        # caught anything — `persisted = True` was set on the very first
        # call every time, the retry never ran, and the
        # record_silent_failure metric below was dead code despite the
        # comment above describing it as implemented. _persist() now
        # returns bool, so the retry is driven off the actual outcome.
        persisted = False
        for attempt in range(2):
            persisted = evicted_graph._persist()
            if persisted:
                break
            if attempt == 0:
                logger.warning(
                    f"Persist attempt 1 failed for evicted graph {evicted_id}, retrying"
                )
        if not persisted:
            # Do NOT drop it. Evicting a graph whose contents are not on
            # disk destroys them outright — the one outcome this product
            # exists to prevent. Memory pressure is a bounded, recoverable
            # cost; losing a session's memory is not. Put it back (as
            # most-recently-used so the next sweep tries a different
            # victim) and stay one entry over cap until the write
            # succeeds or the session is flushed at shutdown.
            _graph_cache[evicted_id] = evicted_graph
            _graph_cache.move_to_end(evicted_id)
            logger.error(
                f"Graph {evicted_id} could NOT be persisted — keeping it in "
                f"memory rather than evicting (cache is over cap by design "
                f"until this write succeeds). Unsaved nodes are still intact."
            )
            _analytics.record_silent_failure("graph_eviction")
            break  # every remaining victim would hit the same DB problem


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

# ── Durability: never lose a session mid-flight ──────────────────────────────
#
# Graph state lives in memory (_graph_cache) between turns and is written
# to SQLite by _persist(). That leaves three windows where a session's
# memory could be lost even though nothing was "broken":
#
#   1. Shutdown. A SIGTERM (docker stop, k8s rollout, systemd restart) ran
#      the lifespan shutdown hook, which logged one line and exited. Every
#      dirty graph still in _graph_cache was dropped unwritten, and any
#      in-flight background extraction task was killed mid-run. A routine
#      deploy silently truncated every active session's memory.
#   2. Hard kill / crash. SIGKILL or OOM gives no shutdown hook at all, so
#      anything not yet persisted is gone. This can't be eliminated, but it
#      can be bounded — see the periodic flusher.
#   3. Eviction with a failing DB — handled in _graph_cache_touch above.
#
# FLUSH_INTERVAL_SECONDS is the worst-case exposure for case 2: a hard
# kill can lose at most this much graph activity.
FLUSH_INTERVAL_SECONDS = 30


async def _flush_all_graphs(reason: str) -> tuple[int, int]:
    """force-persist every cached graph. Returns (flushed, failed).

    force=True because the dirty flag only tracks mutations made through
    add_node()/add_edge(); direct field mutation elsewhere would otherwise
    be skipped, and at shutdown "probably already saved" is not good enough.
    """
    flushed = failed = 0
    for sid, graph in list(_graph_cache.items()):
        try:
            if graph._persist(force=True):
                flushed += 1
            else:
                failed += 1
                _analytics.record_silent_failure("graph_flush")
        except Exception as e:
            failed += 1
            logger.error(f"Flush failed for session {sid} ({reason}): {e}")
            _analytics.record_silent_failure("graph_flush")
    if flushed or failed:
        logger.info(f"Graph flush ({reason}): {flushed} saved, {failed} failed")
    return flushed, failed


async def _periodic_flush() -> None:
    """Bound hard-kill exposure by flushing dirty graphs on a timer."""
    while True:
        try:
            await asyncio.sleep(FLUSH_INTERVAL_SECONDS)
            await _flush_all_graphs("periodic")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # a flush bug must never kill the loop
            logger.error(f"Periodic flush cycle failed: {e}")


async def _drain_background_tasks(timeout: float = 10.0) -> None:
    """Let in-flight background extraction finish before we stop.

    These tasks mutate the graph, so cancelling them outright at shutdown
    would discard work already paid for (including the cheap-model call
    that was already billed).
    """
    pending = [t for t in _background_tasks if not t.done()]
    if not pending:
        return
    logger.info(f"Waiting up to {timeout}s for {len(pending)} background task(s)")
    done, still_pending = await asyncio.wait(pending, timeout=timeout)
    for t in still_pending:
        logger.warning("Background task did not finish before shutdown — cancelling")
        t.cancel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TokenMizer starting")
    flusher = asyncio.create_task(_periodic_flush())
    try:
        yield
    finally:
        # Ordering matters: stop the timer, let background writers finish
        # (they add nodes), and only then flush — otherwise a task
        # completing after the flush would leave its work unwritten.
        flusher.cancel()
        try:
            await flusher
        except asyncio.CancelledError:
            pass
        await _drain_background_tasks()
        await _flush_all_graphs("shutdown")
        logger.info("TokenMizer stopped")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="TokenMizer",
    description="Never lose your AI context again.",
    version="0.4.1",
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


def _rate_limit_key(request: Request) -> str:
    """Identify the client for rate-limiting purposes.

    Default: the TCP peer address, which a caller cannot vary per
    request the way a header can. Behind a reverse proxy that peer is
    the PROXY, so every client collapses into a single shared bucket and
    one heavy user rate-limits everyone else. Operators who terminate
    through a proxy they control can set trust_proxy_headers=true to key
    on the forwarded client address instead.

    X-Forwarded-For is caller-supplied: a client can prepend arbitrary
    entries. Only the entries YOUR proxies appended can be believed, so
    we index from the right by trusted_proxy_hops rather than taking the
    leftmost value (the classic spoofable mistake). With this disabled —
    the default — the header is ignored entirely.
    """
    peer = request.client.host if request.client else "unknown"
    if not getattr(settings, "trust_proxy_headers", False):
        return peer

    forwarded = request.headers.get("X-Forwarded-For", "")
    if not forwarded:
        return peer
    parts = [p.strip() for p in forwarded.split(",") if p.strip()]
    hops = max(1, getattr(settings, "trusted_proxy_hops", 1))
    idx = len(parts) - hops
    if idx < 0:
        # Fewer entries than our own proxies would have added — the chain
        # isn't what we were told to expect, so don't trust any of it.
        return peer
    return parts[idx]


async def _check_rate_limit(request: Request) -> None:
    """
    Raise 429 if client is rate-limited.

    FIXED (TM-05): previously keyed the bucket on
    `request.headers.get("Authorization", client_ip)` — a raw,
    client-supplied string. A client could vary that header per request
    (free in dev mode, where it isn't even validated) and land in a fresh
    bucket every time, never being limited at all. `request.client.host`
    reflects the actual TCP-level source of the connection; it isn't
    something the caller can vary per request the way a header string can.

    This doesn't yet differentiate between individual API keys (the
    current auth model has one shared key for the whole deployment — see
    TM-15), so per-key rate limiting isn't meaningfully possible until
    that lands. IP-based limiting is the correct available control today.
    """
    allowed, retry_after = await _rate_limiter.check(_rate_limit_key(request))
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
    context_window = _context_window(model)

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

                _track_background_task(_background_extract())
            else:
                graph.extract_from_messages(raw_messages, incremental=True)
        else:
            graph.extract_from_messages(raw_messages, incremental=True)
    else:
        graph.extract_from_messages(raw_messages, incremental=True)

    # Smart windowing. `memory.enabled` gates this — it is the switch for
    # the memory subsystem's summarisation behaviour, and until now
    # nothing read it, so turning it off silently changed nothing.
    if settings.memory.enabled and needs_windowing(
        messages, settings.memory.max_tokens_before_summary, model
    ):
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
            else:
                # FIXED (TM-10): previously had no else branch here — a
                # request with no system message did the graph query,
                # built the context block, and threw it away. A system
                # message was only guaranteed to exist because layer 2
                # (terse-output injection) happens to add one, and only
                # when settings.terse_output.enabled is True — a
                # completely unrelated setting. Turning THAT off silently
                # disabled graph context injection too, with no error and
                # no indication anything was skipped.
                messages.insert(0, {
                    "role": "system",
                    "content": f"[Relevant session context]\n{ctx_block}",
                })

    # Context occupancy — measured directly from what will actually be
    # sent to the provider THIS turn (post file-intelligence/compression/
    # windowing/context-injection), not from a stateful accumulator.
    #
    # FIXED (TM-04): this used to be `(context_used + input_tokens) /
    # window`, where context_used was a running total read from a state
    # backend and re-written every call. Two problems, independent of any
    # concurrency race: each `messages` list already contains the FULL
    # running conversation (the OpenAI-style contract), so the
    # accumulator double-counted every earlier turn's content on top of
    # itself each subsequent turn — diverging further from real occupancy
    # the longer a session ran. And it never reflected windowing: once
    # the tracked value crossed trigger_at_percent it never came back
    # down, so every later turn re-triggered a full checkpoint write
    # (mitigated separately by the auto-retention cap below, but the root
    # cause was the accumulator itself). Measuring the actual outgoing
    # payload here is a pure function of `messages` — no shared mutable
    # state, so nothing to race on, and it naturally reflects windowing
    # having just shrunk `messages` above.
    context_pct = count_messages_tokens(messages, model) / context_window

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
    #
    # Retention/frequency: CheckpointManager caps how many auto-triggered
    # checkpoints are kept per session (oldest pruned first, manual ones
    # never touched — see checkpoints/manager.py::_prune_auto_checkpoints).
    # That bounds storage growth for a session that stays near the
    # threshold for many turns; it intentionally does not suppress this
    # turn's checkpoint attempt, since each attempt reflects this turn's
    # real (non-accumulated) occupancy rather than noise.
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

    return messages, checkpoint_status


async def _call_provider(
    req,
    messages: list[dict],
    model: str,
    user_content: str,
    session_id: str,
    savings: dict,
) -> tuple[str, int, int, float, bool]:
    """
    Layer 3 + 5: Cache lookup → LLM call → output trim → cache write.
    Returns (response_text, input_tokens, output_tokens, latency_ms, cache_hit).

    FIXED (TM-34): cache_hit is now returned explicitly instead of being
    inferred downstream from `input_tokens_actual == 0` — that inference
    would misclassify any real provider response that happens to report
    zero input tokens as a cache hit. This function already knows exactly
    which path it took; it should just say so.
    """
    # Cache lookup
    if settings.cache.enabled and user_content:
        cached = _cache.get(user_content, session_id=session_id)
        if cached:
            savings["cache"] = count_tokens(user_content, model)
            output_tokens = count_tokens(cached.response, model)
            return cached.response, 0, output_tokens, 0.0, True

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
        # FIXED (TM-33): previously echoed str(e) verbatim into the 502
        # response body. Provider SDK exceptions routinely embed request
        # URLs, query params, or other internal detail that shouldn't
        # reach an API client. Full detail is logged server-side (with a
        # correlation id so it can be found), the client gets a generic
        # message plus that same id.
        correlation_id = uuid.uuid4().hex[:12]
        logger.error(f"Provider error [{correlation_id}]: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Provider request failed (ref: {correlation_id}). "
                   f"Check server logs for details.",
        )

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

    return response_text, input_tokens, output_tokens, latency_ms, False


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
        stream_failed = False
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
            stream_failed = True
            yield "data: " + _json.dumps(
                {"error": {"message": str(e), "type": "provider_error"}}
            ) + "\n\n"
        except Exception as e:
            # Anything the provider layer didn't wrap in ProviderError
            # (transport errors, decode errors, bugs) used to escape the
            # generator entirely: the SSE stream died with no terminator
            # and the bookkeeping below never ran. Same treatment — tell
            # the client, then finish cleanly.
            stream_failed = True
            correlation_id = uuid.uuid4().hex[:12]
            logger.error(f"Stream error [{correlation_id}]: {e}")
            yield "data: " + _json.dumps(
                {"error": {"message": f"Stream failed (ref: {correlation_id})",
                           "type": "internal_error"}}
            ) + "\n\n"
        yield _chunk({}, finish="stop")
        yield "data: [DONE]\n\n"

        # Post-stream bookkeeping
        latency_ms = (time.monotonic() - t0) * 1000
        output_tokens = count_tokens(full_text, model)
        input_tokens = count_messages_tokens(messages, model)
        # Never cache a response that didn't finish. `full_text` after a
        # mid-stream failure holds however many tokens arrived before the
        # error — writing that to the cache would serve a silently
        # truncated answer to every future matching prompt, long after the
        # provider recovered. Re-writing a cache HIT is equally pointless.
        if (settings.cache.enabled and user_content and full_text
                and not stream_failed and not cache_hit):
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

    # Bind the session to this caller (or verify an existing binding).
    # session_id is client-supplied, so without this any caller could
    # name someone else's session and have their conversation folded
    # into that session's graph — a write-side version of the same hole
    # the read endpoints had. See security/ownership.py.
    principal = getattr(request.state, "principal", DEV_PRINCIPAL)
    try:
        _ownership.check_access(session_id, principal, claim=True)
    except SessionAccessDenied:
        logger.warning(f"Denied chat request for session {session_id!r} — different principal")
        raise HTTPException(
            status_code=403,
            detail=f"Session '{session_id}' belongs to a different API key. "
                   f"Use a different session_id, or omit it for a new session.",
        )
    except OwnershipUnavailable as e:
        logger.error(f"Ownership store unavailable, denying chat request: {e}")
        raise HTTPException(
            status_code=503,
            detail="Session ownership state unavailable — request rejected for safety.",
        )

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

    orig_input_tokens = count_messages_tokens(raw_messages, model)
    savings["routing"] = 0

    # Layer 4: graph update + context injection (mutates messages)
    checkpoint_status: dict = {"attempted": False, "succeeded": False, "checkpoint_id": None}
    if settings.graph_checkpoint.enabled:
        graph    = await _get_graph_async(session_id)
        # Hold the session in-use for as long as we're mutating its graph,
        # so a concurrent request's cache eviction can't force-persist and
        # detach this instance out from under us (which would silently
        # drop everything added below).
        with _session_in_use(session_id):
            messages, checkpoint_status = await _update_graph(
                session_id, graph, raw_messages, messages, model, savings, user_query
            )

    # FIXED (TM-11): input_tokens_sent used to be measured HERE, before
    # _update_graph() ran — so it never reflected either the reduction
    # from windowing or the addition from graph-context injection, both
    # of which happen inside _update_graph(). Measuring it after means
    # this number (reported in analytics/dashboard as "tokens actually
    # sent") is the actual size of what's about to be sent to the
    # provider, not a stale pre-windowing estimate.
    sent_input_tokens = count_messages_tokens(messages, model)

    # Streaming: true SSE passthrough (v0.3). Output-trimming is skipped in
    # stream mode (can't trim tokens that already left the building) — all
    # input-side layers (file intel, compression, graph context) still apply.
    if req.stream:
        return _stream_response(req, messages, model, user_content,
                                session_id, savings, orig_input_tokens)

    # Layer 5: call provider (or return cache hit)
    response_text, input_tokens_actual, output_tokens, latency_ms, cache_hit = await _call_provider(
        req, messages, model, user_content, session_id, savings
    )

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
# Graph inspection, checkpoint, and decision-management endpoints live in
# routes_graph.py (split out to keep this file focused on the core proxy
# path). Imported at the bottom of this module — by this point every
# singleton/helper routes_graph.py references via `app_module.<name>`
# (_analytics, _cache, _checkpoint_mgr, _get_graph_async, _check_rate_limit)
# is already defined above.
from tokenmizer.api.routes_graph import router as _graph_router  # noqa: E402

app.include_router(_graph_router)
