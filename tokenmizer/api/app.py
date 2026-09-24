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
import contextlib
import logging
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager, contextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from tokenmizer import __version__
from tokenmizer.analytics.engine import AnalyticsEngine
from tokenmizer.api.rate_limiter import get_rate_limiter
from tokenmizer.checkpoints.manager import CheckpointManager
from tokenmizer.compression.engine import CompressionPipeline
from tokenmizer.compression.output_trimmer import OutputTrimmer
from tokenmizer.compression.window import SmartMessageWindow, needs_windowing
from tokenmizer.config.settings import get_settings, resolve_semantic_retrieval
from tokenmizer.core.tokenizer import count_messages_tokens, count_tokens
from tokenmizer.filters.file_intelligence import FileIntelligence
from tokenmizer.graph_memory.graph import GraphMemory
from tokenmizer.providers.providers import build_provider
from tokenmizer.security.auth import verify_api_key
from tokenmizer.security.fencing import fence
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
    Best-effort startup notice for multi-worker deployments (e.g.
    `uvicorn --workers 4`) sharing one storage_dir.

    Graph writes ARE safe across processes: per-row storage plus the
    cross-process file lock in graph_memory/filelock.py make persist a
    locked read-modify-write, so concurrent writers merge and a stale
    writer adopts another's deletions instead of reinstating them.
    Measured lossless with 4 processes writing one session
    (benchmarks/persistence/runner.py).

    What is still per-process, and therefore divergent per worker:
    analytics counters, rate-limit buckets, and the semantic cache. Each
    worker also keeps its own `_graph_cache`, so a session's in-memory
    copy can be briefly stale between writes.

    Not detectable with certainty from inside one process (there is no
    universal "how many workers am I one of" signal), so this checks
    common launcher env vars. A false negative is expected for unusual
    launch setups; the goal is to surface the caveats in the common case.
    """
    import os
    for var in ("WEB_CONCURRENCY", "UVICORN_WORKERS", "GUNICORN_WORKERS"):
        val = os.environ.get(var, "").strip()
        if val.isdigit() and int(val) > 1:
            logger.warning(
                f"{var}={val} suggests multiple worker processes. Graph "
                "and checkpoint writes are cross-process safe, but "
                "analytics, rate limiting and the semantic cache are "
                "per-process, so those will differ per worker. Rate "
                "limits in particular apply PER WORKER, i.e. the "
                "effective limit is roughly N times what you configured. "
                "Note also that flock is unreliable on NFS — keep "
                "storage_dir on a local filesystem."
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
    max_bytes=settings.cache.max_bytes,
    max_entry_bytes=settings.cache.max_entry_bytes,
    max_semantic_scan=settings.cache.max_semantic_scan,
)
_checkpoint_mgr = CheckpointManager(
    storage_dir=settings.graph_checkpoint.storage_dir,
    max_resume_tokens=settings.graph_checkpoint.max_resume_tokens,
)
_ownership = OwnershipStore(storage_dir=settings.graph_checkpoint.storage_dir)
# storage_dir gives analytics the same durability the graph has. Without
# it every savings figure resets to zero on restart — see engine.py.
_analytics = AnalyticsEngine(storage_dir=settings.graph_checkpoint.storage_dir)
# Built only when asked for: the store creates a database file, and a
# feature that is off should leave no trace on disk.
# Resolved once, at import: "auto" asks whether the embedding model
# actually loads, which is a question worth asking at startup and not on
# every request. See resolve_semantic_retrieval.
_SEMANTIC_RETRIEVAL = resolve_semantic_retrieval(
    settings.graph_checkpoint.semantic_retrieval)
if _SEMANTIC_RETRIEVAL and settings.graph_checkpoint.semantic_retrieval == "auto":
    logger.info("Semantic retrieval is ON — the embedding model loaded. "
                "Set graph_checkpoint.semantic_retrieval: false to pin it off.")

_preferences = None
if settings.preferences.enabled:
    from tokenmizer.preferences import PreferenceStore
    _preferences = PreferenceStore(
        storage_dir=settings.graph_checkpoint.storage_dir)
_output_trimmer = OutputTrimmer()


def _build_rate_limiter():
    """The shared limiter when the deployment asked for one, else the
    per-process one.

    `state_backend: memory` (the default) enforces the configured limit
    once per worker, which with `--workers 4` is four times the number in
    the config. That is fine for one process and wrong for the deployment
    the Dockerfile ships, so `sqlite` puts the buckets where every worker
    on the host can see them. A shared store that fails to open falls back
    here rather than failing the proxy — the old behaviour, loudly.
    """
    if settings.state_backend == "sqlite":
        from tokenmizer.api.shared_rate_limiter import SQLiteRateLimiter
        shared = SQLiteRateLimiter(
            rate=60, per_seconds=60, burst=10,
            storage_dir=settings.graph_checkpoint.storage_dir,
        )
        if shared.available:
            return shared
    return get_rate_limiter(rate=60, per_seconds=60, burst=10)


_rate_limiter = _build_rate_limiter()

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
# unbounded *over time* — but nothing bounded how many could be in flight
# AT ONCE. One burst of a few thousand requests meant a few thousand
# concurrent extraction tasks, each holding its slice of the transcript
# and each making an upstream call: a memory spike and a self-inflicted
# rate limit on the cheap provider, at exactly the moment the proxy is
# busiest. See _extraction_slot() below.
_background_tasks: set[asyncio.Task] = set()

# Backpressure on the background LLM extraction pass.
#
# Shedding this is safe in a way that shedding most work is not: the
# HEURISTIC extraction already ran synchronously before the task was
# scheduled, so the graph has this turn's facts either way. What is lost
# is the accuracy of the LLM pass for that one turn — precisely the
# degradation the existing exception handler already documents as "no
# data lost, just less accurate extraction this turn".
#
# Concurrency, not a queue length, is the limit that matters: a queue
# just moves the memory from tasks to a list and adds latency to work
# that is already best-effort. Over the limit, the turn keeps its
# heuristic facts and the shed is counted in /api/stats.
_EXTRACTION_MAX_CONCURRENT = 8
_extraction_inflight = 0
# Tasks created but not yet started. The slot counter alone is not enough:
# a task that has not been scheduled yet holds its closure — including the
# whole transcript it was given — so creating N of them and letting them
# shed on entry still spends the memory the cap exists to save. Capacity
# is therefore checked BEFORE the task is created, and again on entry,
# because the loop can hand out slots between the two.
_extraction_pending = 0


def _track_background_task(coro) -> asyncio.Task:
    """asyncio.create_task() + strong-reference retention in one call —
    use this instead of a bare create_task() for any fire-and-forget task
    that must not be garbage-collected before it finishes."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _extraction_has_capacity() -> bool:
    """Is it worth CREATING an extraction task at all?

    Counts the ones already queued as well as the ones running, because a
    queued task is not free: it pins its slice of the transcript until the
    loop gets to it.
    """
    return (_extraction_inflight + _extraction_pending) < _EXTRACTION_MAX_CONCURRENT


@contextlib.contextmanager
def _extraction_slot():
    """Occupy one of the concurrent-extraction slots, or yield False.

    A counter rather than an asyncio.Semaphore because the caller must be
    able to DECLINE when the limit is reached. `Semaphore.acquire()`
    waits, and waiting is the failure mode this exists to prevent: the
    tasks pile up holding memory instead of being dropped cheaply.
    """
    global _extraction_inflight
    if _extraction_inflight >= _EXTRACTION_MAX_CONCURRENT:
        yield False
        return
    _extraction_inflight += 1
    try:
        yield True
    finally:
        _extraction_inflight -= 1


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
        # Evict the oldest unheld lock (OrderedDict order is LRU order).
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
_extraction_provider = None   # lazy — only built if use_llm_extraction=True
# Reason LLM extraction is unavailable, logged ONCE. The check re-runs on
# every request (a key can be configured later without a restart), but the
# warning must not: with use_llm_extraction on and no key it was emitted
# for every chat turn of every session, which buried every other log line.
_extraction_unavailable_reason: Optional[str] = None


def _get_extraction_provider():
    """
    The provider behind LLM extraction: the same provider and model the
    operator configured for chat, unless `graph_checkpoint.extraction_model`
    pins a different model on that provider.

    This used to pick a "cheap" model from a hardcoded per-vendor list.
    Three providers had an entry; the rest silently fell back to heuristic
    extraction with a key configured and use_llm_extraction on. The list
    also encoded a vendor opinion — a smaller model than the one the
    operator had already judged good enough for their answers — and
    extraction is where hallucinated facts would enter memory, so it is
    the last place to quietly trade quality for cost. Reusing the chat
    provider means every adapter works, the key is already there, and
    the operator who wants a cheaper model says so explicitly.

    Only instantiated when use_llm_extraction=True; None means
    "heuristic extraction only", logged once with the reason.
    """
    global _extraction_provider, _extraction_unavailable_reason
    if _extraction_provider is not None:
        return _extraction_provider

    provider = settings.provider.lower()
    override = (settings.graph_checkpoint.extraction_model or "").strip()
    if provider != "ollama" and not settings.get_api_key_for_provider(provider):
        reason = (f"use_llm_extraction is on but provider={provider!r} has no API "
                  f"key configured; falling back to heuristic extraction.")
        if reason != _extraction_unavailable_reason:
            logger.warning(reason)
            _extraction_unavailable_reason = reason
        return None
    try:
        _extraction_provider = build_provider(settings, model=override or None)
    except ValueError as e:
        reason = (f"use_llm_extraction is on but no provider could be built ({e}); "
                  f"falling back to heuristic extraction.")
        if reason != _extraction_unavailable_reason:
            logger.warning(reason)
            _extraction_unavailable_reason = reason
        return None
    _extraction_unavailable_reason = None
    return _extraction_provider


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
        # Flush pending writes before dropping from memory. One retry
        # covers transient SQLite WAL lock contention; the outcome is
        # driven off _persist()'s bool return, and a failure is recorded
        # to analytics so it is queryable via /api/stats.
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
                semantic_retrieval=_SEMANTIC_RETRIEVAL,
            )
        _graph_cache_touch(session_id)
        return _graph_cache[session_id]


# ── Context window sizes ──────────────────────────────────────────────────────

# Context window per model family, in tokens. Only feeds the auto-checkpoint
# trigger (context_pct = tokens sent / window), so an entry that is too
# SMALL checkpoints early (harmless) and one that is too LARGE never
# checkpoints (the failure this table exists to prevent) — when unsure,
# prefer the smaller published figure. Longest matching key wins, so a
# specific entry beats its family's catch-all. Newest Claude models
# (fable-5, opus-4-8, sonnet-5, haiku-4-5) all match the "claude" entry;
# add a specific entry ONLY if a model's window differs.
_CONTEXT_WINDOWS = {
    "claude-fable-5": 200_000, "claude-opus-4-8": 200_000,
    "claude-sonnet": 200_000, "claude-opus": 200_000, "claude-haiku": 200_000,
    "claude": 200_000,
    # OpenAI: the 4.1 family and the 5 series are far larger than 4o; the
    # reasoning models sit at 200k. Longest-key matching keeps "gpt-4o"
    # from being shadowed by "gpt-4", and "gpt-4.1" from matching "gpt-4".
    "gpt-4.1": 1_000_000, "gpt-4o": 128_000, "gpt-4": 128_000, "gpt-3.5": 16_000,
    "gpt-5": 400_000,
    "o1": 200_000, "o3": 200_000, "o4": 200_000,
    "gemini": 1_000_000,
    "deepseek": 128_000,
    "mistral": 128_000, "codestral": 256_000,
    "grok": 128_000,
    "command-r": 128_000,
    # Local models vary by build; 32k is the common default `num_ctx`
    # ceiling, and a too-small figure only checkpoints early.
    "llama": 32_000, "qwen": 32_000, "mixtral": 32_000, "phi": 16_000,
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
    """Bound hard-kill exposure by flushing dirty graphs on a timer, and
    keep the per-session lock directory from growing without bound."""
    cycles = 0
    while True:
        try:
            await asyncio.sleep(FLUSH_INTERVAL_SECONDS)
            await _flush_all_graphs("periodic")
            # Same timer, same worst-case exposure: a hard kill loses at
            # most one interval of analytics, not the whole history.
            if not _analytics.flush():
                _analytics.record_silent_failure("analytics_flush")
            cycles += 1
            # Roughly hourly at the default 30s interval. Lock files are
            # empty but one is created per session touched and never
            # removed on release, so without this the directory gains an
            # inode per session forever.
            if cycles % 120 == 0:
                from tokenmizer.graph_memory.filelock import sweep_stale_locks
                sweep_stale_locks(settings.graph_checkpoint.storage_dir)
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
        _analytics.flush()
        logger.info("TokenMizer stopped")


# ── App ───────────────────────────────────────────────────────────────────────

# Derived, never literal. A hardcoded version here is a string nobody
# thinks to update at release time: it drifted for four releases before a
# consistency test caught it, and in the meantime /docs and the OpenAPI
# schema advertised a version the package had not been on for months.
# The data files that cannot import Python (server.json, plugin.json,
# marketplace.json) still pin literals, and tests/unit/test_version_
# consistency.py holds those to __version__.
app = FastAPI(
    title="TokenMizer",
    description="Never lose your AI context again.",
    version=__version__,
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
    their text parts preserved.

    Tool calling carries three more fields, all optional: an assistant
    turn's `tool_calls`, and a `role: "tool"` turn's `tool_call_id` (and
    OpenAI's optional `name`). They ride through the pipeline untouched
    and reach the provider adapter, which speaks them natively or
    translates them (see providers/tools.py)."""
    model_config = {"extra": "allow"}

    role: str
    content: str | list | None = ""
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def text(self) -> str:
        from tokenmizer.graph_memory.helpers import _content_to_text
        return _content_to_text(self.content)

    def to_dict(self) -> dict:
        """The pipeline's message shape: role + text content, plus the
        tool fields only when set, so an ordinary message is exactly the
        two-key dict every layer has always seen."""
        d: dict = {"role": self.role, "content": self.text()}
        if self.tool_calls:
            from tokenmizer.providers.tools import normalize_tool_calls
            d["tool_calls"] = normalize_tool_calls(self.tool_calls)
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d


class ChatRequest(BaseModel):
    """OpenAI-compatible request. Sampling params (temperature, top_p, stop)
    are forwarded to the provider. Unknown fields are accepted and ignored
    (extra='allow') so standard OpenAI clients never get a 422 — but only
    the fields below influence the call."""
    model_config = {"extra": "allow"}

    model: Optional[str] = None
    messages: list[ChatMessage]
    # Ranges every provider enforces anyway. Checked here so a bad value is
    # a 422 naming the field, not an opaque upstream 400 after the request
    # has been compressed, extracted and forwarded — `max_tokens: -5` went
    # all the way to the provider.
    max_tokens: Optional[int] = Field(default=4096, ge=1)
    # OpenAI's current name for the same limit; newer SDK defaults and the
    # o-series reject `max_tokens` and send this instead. Without it the
    # client's limit was silently replaced by the 4096 default above.
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    stream: Optional[bool] = False
    session_id: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    stop: Optional[str | list[str]] = None
    # Tool/function calling, OpenAI shape. Forwarded to providers that
    # support it (see BaseProvider.supports_tools); a 501 otherwise.
    tools: Optional[list[dict]] = None
    tool_choice: Optional[str | dict] = None
    parallel_tool_calls: Optional[bool] = None


def _max_tokens(req: "ChatRequest") -> int:
    """The completion limit the client asked for, under either name."""
    if req.max_completion_tokens is not None:
        return req.max_completion_tokens
    return req.max_tokens or 4096


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


def _tool_kwargs(req: "ChatRequest") -> dict:
    """Tool-calling params to forward (only when the request declares tools)."""
    if not req.tools:
        return {}
    kw: dict = {"tools": req.tools}
    if req.tool_choice is not None:
        kw["tool_choice"] = req.tool_choice
    if req.parallel_tool_calls is not None:
        kw["parallel_tool_calls"] = req.parallel_tool_calls
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

    Buckets key on the connection source, never on a client-supplied
    header: a caller who can vary the key can mint a fresh bucket per
    request and is never limited. See _rate_limit_key() for the
    trusted-proxy exception.

    Limits are per-address, not per-API-key. Splitting by key would only
    be meaningful for deployments that configure several (see
    settings.api_keys); addresses are the control that always applies.
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
        terse = _compression.terse_system_prompt(
            settings.terse_output.level, style=settings.terse_output.style)
        has_system = any(m.get("role") == "system" for m in messages)
        if has_system:
            for m in messages:
                if m.get("role") == "system":
                    m["content"] = terse + "\n\n" + m["content"]
                    break
        else:
            messages = [{"role": "system", "content": terse}] + messages

    return messages


def _last_substantive_query(raw_messages: list[dict], min_words: int = 4) -> str:
    """The most recent user turn long enough to retrieve against.

    A follow-up like "why?" carries its subject in the turn before it. Walks
    backwards past the short turns rather than giving up on retrieval, so the
    graph is still searched with the topic actually under discussion.
    """
    for message in reversed(raw_messages):
        if message.get("role") != "user":
            continue
        content = message.get("content") or ""
        if len(content.split()) >= min_words:
            return content
    return ""


async def _cross_session_query(
    session_id: str, graph, principal: str, retrieval_query: str, top_k: int,
) -> list[tuple]:
    """Rank `retrieval_query` against `graph` plus this principal's other
    sessions. Other sessions are loaded through _get_graph_async, so they
    share the same LRU cache and eviction/lock behaviour as any other
    session the proxy touches — a cross-session read is not a special
    code path, just another cache hit or miss.
    """
    from tokenmizer.graph_memory.cross_session import (
        MAX_OTHER_SESSIONS,
        query_across_sessions,
    )

    try:
        other_ids = [sid for sid in _ownership.sessions_for(principal) if sid != session_id]
    except OwnershipUnavailable:
        other_ids = []
    other_ids = other_ids[:MAX_OTHER_SESSIONS]

    other_graphs: dict[str, GraphMemory] = {}
    for sid in other_ids:
        try:
            other_graphs[sid] = await _get_graph_async(sid)
        except Exception as e:
            logger.warning(f"Cross-session recall: could not load session {sid!r}: {e}")

    return query_across_sessions(
        graph, list(other_graphs), retrieval_query, top_k,
        load_graph=lambda sid: other_graphs[sid],
    )


async def _update_graph(
    session_id: str,
    graph,
    raw_messages: list[dict],
    messages: list[dict],
    model: str,
    savings: dict,
    user_query: str,
    principal: str = DEV_PRINCIPAL,
) -> tuple[list[dict], dict]:
    """
    Layer 4: Graph extraction, smart windowing, context injection, checkpoint.
    Mutates messages (adds graph context).
    Returns (updated_messages, checkpoint_status) — checkpoint_status surfaces
    auto-checkpoint success/failure to the caller instead of only logging it.
    """
    global _extraction_pending
    context_window = _context_window(model)

    # Extraction: heuristic sync now, LLM async in background
    if settings.graph_checkpoint.use_llm_extraction:
        cheap = _get_extraction_provider()
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
                    global _extraction_pending
                    _extraction_pending -= 1
                    with _extraction_slot() as got_slot:
                        if not got_slot:
                            # At capacity. This turn keeps the heuristic
                            # facts extracted synchronously above; only
                            # the LLM refinement is dropped. Counted as a
                            # shed, not a failure: the operator's fix is
                            # to raise the limit or add capacity, which is
                            # a different action from fixing a broken key.
                            _analytics.record_shed("llm_extraction")
                            logger.info(
                                "Background LLM extraction shed for session "
                                "%s — %d already in flight (limit %d). The "
                                "turn keeps its heuristic facts.",
                                _sid, _extraction_inflight,
                                _EXTRACTION_MAX_CONCURRENT,
                            )
                            return
                        await _run_extraction(_g, _msgs, _all, _cheap, _lock, _sid)

                async def _run_extraction(_g, _msgs, _all, _cheap, _lock, _sid):
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
                            # Warning, not debug: this path can fail on
                            # every call for the whole session (expired
                            # cheap-provider key, outage, exhausted quota)
                            # while the graph just quietly stops gaining
                            # nodes. Also counted in analytics so repeated
                            # failures show up in /api/stats.
                            logger.warning(
                                f"Background LLM extraction failed for session "
                                f"{_sid} (falling back to heuristic-only on next "
                                f"calls, no data lost — just less accurate "
                                f"extraction this turn): {e}"
                            )
                            _analytics.record_silent_failure("llm_extraction")

                if _extraction_has_capacity():
                    _extraction_pending += 1
                    _track_background_task(_background_extract())
                else:
                    # Do not even build the coroutine: its closure holds
                    # this turn's transcript, and a queue of those is the
                    # memory spike the cap exists to prevent.
                    _analytics.record_shed("llm_extraction")
                    logger.info(
                        "Background LLM extraction not scheduled for session "
                        "%s — %d running, %d queued (limit %d). The turn keeps "
                        "its heuristic facts.",
                        session_id, _extraction_inflight, _extraction_pending,
                        _EXTRACTION_MAX_CONCURRENT,
                    )
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

    # Context injection — only when graph has enough signal.
    #
    # There used to be a `len(user_query.split()) >= 4` condition here, so a
    # short turn got no memory at all: "why?", "continue", "fix it", "same as
    # before" and "run the tests" were all below the bar. Those are exactly
    # the turns where the model most needs to be told what "it" and "before"
    # refer to — the graph held the answer and the proxy declined to pass it
    # on, on a word count.
    #
    # A short turn is not a turn with no topic; it is a turn whose topic is
    # in the previous one. So retrieve against the last substantive user
    # message instead of skipping retrieval, and let relevance decide.
    retrieval_query = user_query
    if len(user_query.split()) < 4:
        retrieval_query = _last_substantive_query(raw_messages) or user_query

    if len(graph._nodes) >= 3 and retrieval_query.strip():
        if settings.graph_checkpoint.cross_session_recall:
            relevant_pairs = await _cross_session_query(
                session_id, graph, principal, retrieval_query, top_k=8
            )
        else:
            relevant_pairs = [(n, session_id) for n in graph.query(retrieval_query, top_k=8)]
        if relevant_pairs:
            ctx_parts = [
                f"  {n.type.value}: {n.label}"
                + (f" ({n.summary[:50]})" if n.summary else "")
                # Tag only nodes recalled from a DIFFERENT session — the
                # common case (feature off, or this session's own nodes
                # won the ranking) keeps the exact wording the model has
                # always seen.
                + (f" [from session {sid}]" if sid != session_id else "")
                for n, sid in relevant_pairs[:6]
            ]
            ctx_block = "\n".join(ctx_parts)
            sys_idx = next(
                (i for i, m in enumerate(messages) if m.get("role") == "system"), None
            )
            if sys_idx is not None:
                # APPENDED, not prepended. Provider prompt caching (layer
                # 5) caches the longest unchanged prefix of the system
                # prompt; this block changes every turn, and at the front
                # it invalidated the cache on every request behind any
                # agent with a long stable system prompt — the one place
                # the cache pays. At the end, the terse prompt and the
                # client's own system prompt stay cacheable.
                messages[sys_idx]["content"] = (
                    f"{messages[sys_idx]['content']}\n\n"
                    f"{fence(ctx_block, 'relevant session context')}"
                )
            else:
                # A system message is not guaranteed to exist: layer 2
                # only adds one when terse_output is enabled. Without
                # this branch, graph context injection would silently
                # depend on that unrelated setting.
                messages.insert(0, {
                    "role": "system",
                    "content": fence(ctx_block, "relevant session context"),
                })

    # Preferences — habits that outlive this session, and so are NOT in
    # this session's graph: "keep it brief", "always TypeScript". Off
    # unless asked for; see PreferenceSettings for why. Appended for the
    # same prompt-caching reason as the block above, and after it, since
    # what this session is about matters more than how the reader likes
    # their answers formatted.
    if settings.preferences.enabled and _preferences is not None:
        try:
            if user_query:
                _preferences.observe(principal, user_query)
            pref_block = _preferences.context(
                principal,
                max_items=settings.preferences.max_items,
                max_chars=settings.preferences.max_chars,
            )
        except Exception as e:                       # pragma: no cover - defensive
            logger.warning("Preferences skipped for this turn: %s", e)
            pref_block = ""
        if pref_block:
            sys_idx = next(
                (i for i, m in enumerate(messages) if m.get("role") == "system"), None
            )
            fenced = fence(pref_block, "remembered preferences")
            if sys_idx is not None:
                messages[sys_idx]["content"] = (
                    f"{messages[sys_idx]['content']}\n\n{fenced}")
            else:
                messages.insert(0, {"role": "system", "content": fenced})

    # Context occupancy, measured per turn rather than accumulated: each
    # `messages` list already carries the full running conversation, so a
    # running total would double-count every earlier turn and could never
    # fall again after windowing shrank the payload. A pure function of
    # its inputs — no shared mutable state.
    #
    # Measured against BOTH sides, and the fuller one wins:
    #
    #   sent — what leaves here this turn, after file intelligence,
    #          compression, windowing and injection. This is what the
    #          provider's window has to hold.
    #   raw  — the conversation the CLIENT is holding, which is what is
    #          actually running out of room and what a resume has to
    #          replace.
    #
    # Comparing only `sent` made the trigger almost inert exactly where it
    # matters: windowing keeps the sent payload near-constant, so a session
    # 40 turns deep sent the same 30% it sent at turn 5 and never
    # checkpointed, while the client's own history was the thing about to
    # be truncated. Now that resume reads the live graph, nothing is lost
    # when the trigger is late — but the checkpoint diff and the "Continue
    # from" hint are, and those are the parts a resume cannot rebuild.
    sent_pct = count_messages_tokens(messages, model) / context_window
    raw_pct = count_messages_tokens(raw_messages, model) / context_window
    context_pct = max(sent_pct, raw_pct)

    # Auto-checkpoint.
    #
    # A failed checkpoint must NOT fail the chat request — the caller came
    # for an answer — but it must be visible, so the outcome is returned
    # in the `tokenmizer.checkpoint` response field rather than only
    # logged. One retry covers transient SQLite lock contention under
    # concurrent requests.
    #
    # Frequency is not throttled here: each attempt reflects this turn's
    # real occupancy. Storage growth is bounded instead by
    # CheckpointManager._prune_auto_checkpoints, which caps retained
    # auto-checkpoints per session and never touches manual ones.
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
    raw_messages: Optional[list[dict]] = None,
    outcome: Optional[dict] = None,
) -> tuple[str, int, int, float, bool]:
    """
    Layer 3 + 5: Cache lookup → LLM call → output trim → cache write.
    Returns (response_text, input_tokens, output_tokens, latency_ms, cache_hit).

    cache_hit is returned explicitly rather than inferred downstream from
    `input_tokens == 0`, which would misclassify any real provider
    response that happens to report zero input tokens.
    """
    # Cache lookup. Keyed on the prompt AND the conversation it was asked
    # in — see SemanticCache.conversation_fingerprint.
    cache_ctx = _cache.conversation_fingerprint(raw_messages or [])
    if settings.cache.enabled and user_content:
        cached = _cache.get(user_content, session_id=session_id, context=cache_ctx)
        if cached:
            savings["cache"] = count_tokens(user_content, model)
            output_tokens = count_tokens(cached.response, model)
            return cached.response, 0, output_tokens, 0.0, True

    # LLM call. `messages` is already redacted — redaction happens once
    # at ingestion in chat_completions(), so every downstream consumer
    # shares one safe copy. Deliberately NOT re-redacted here: doing so
    # would mask a regression if ingestion ever stopped redacting.
    provider = _get_provider()
    kwargs = dict(model=model, max_tokens=_max_tokens(req), stream=False,
                  **_sampling_kwargs(req), **_tool_kwargs(req))
    try:
        resp = await provider.chat(messages=messages, **kwargs)
    except Exception as e:
        # The request that failed is the one TokenMizer BUILT — windowed,
        # compressed, with context injected. If it differs from what the
        # client actually sent, the fault may be in a transform rather than
        # in the provider, and a session must not die on that: before this,
        # a windowing bug that produced an assistant-first message list
        # turned into a 502 on every remaining turn of every long session,
        # with nothing pointing at the cause. So try once more with the
        # client's own messages (already redacted). One extra provider
        # call, only on a path that was returning an error anyway. If it
        # succeeds the turn goes through at full price, the savings for
        # this turn are zero, and the response says so; if it also fails,
        # the 502 below stands.
        correlation_id = uuid.uuid4().hex[:12]
        if raw_messages is not None and raw_messages != messages:
            logger.warning(
                f"Provider rejected the transformed request [{correlation_id}] "
                f"for session {session_id!r}; retrying with the untransformed "
                f"messages: {type(e).__name__}"
            )
            try:
                resp = await provider.chat(messages=raw_messages, **kwargs)
            except Exception as e2:
                logger.error(f"Provider error [{correlation_id}] (both attempts): "
                             f"transformed={e!r}; untransformed={e2!r}")
                raise HTTPException(
                    status_code=502,
                    detail=f"Provider request failed (ref: {correlation_id}). "
                           f"Check server logs for details.",
                )
            else:
                # The transforms were the problem. That is a TokenMizer bug
                # and it must be visible, not absorbed: counted as a silent
                # failure, and reported on the response.
                logger.error(
                    f"TokenMizer's transformed request was rejected but the "
                    f"untransformed one succeeded [{correlation_id}] — a "
                    f"pipeline layer produced a request the provider will not "
                    f"accept. Original error: {e}"
                )
                _analytics.record_silent_failure("transform_rejected")
                savings.clear()
                if outcome is not None:
                    outcome["transform_rejected"] = True
                    outcome["ref"] = correlation_id
        else:
            # Provider SDK exceptions routinely embed request URLs, query
            # params and other internal detail, so str(e) must not reach
            # the client. Full detail goes to the log under a correlation
            # id the client is given to quote.
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
    if outcome is not None:
        outcome["tool_calls"] = list(resp.tool_calls or [])
        outcome["finish_reason"] = resp.finish_reason

    # A turn the model answered with a tool call is neither trimmed (the
    # text, if any, is the model's note to the caller) nor cached (the
    # cached answer would replay a tool request against a different world).
    if resp.tool_calls:
        return response_text, input_tokens, output_tokens, latency_ms, False

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
                   session_id=session_id, context=cache_ctx)

    return response_text, input_tokens, output_tokens, latency_ms, False


def _stream_response(req, messages, model, user_content, session_id,
                     savings, orig_input_tokens, raw_messages=None) -> StreamingResponse:
    """True SSE passthrough (OpenAI chat.completion.chunk format).

    Cache hits stream as a single chunk. After the stream closes, analytics
    and the semantic-cache write run on the accumulated text — same
    bookkeeping as the non-stream path, minus output trimming.

    Transform-rejected fallback, matching _call_provider(): if the
    transformed `messages` fail before any content has reached the
    client, retry once with `raw_messages`. Once content has streamed,
    a failure can never retry — the client already has an unrecoverable
    partial answer, and resending would duplicate or interleave output —
    so it is always the existing mid-stream error event.
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

    def _chunk(delta: dict, finish: str | None = None, extra: dict | None = None) -> str:
        payload = {
            "id": resp_id, "object": "chat.completion.chunk",
            "created": created, "model": model, "session_id": session_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if extra:
            payload.update(extra)
        return "data: " + _json.dumps(payload) + "\n\n"

    cache_ctx = _cache.conversation_fingerprint(raw_messages or [])

    tool_kw = _tool_kwargs(req)
    # An adapter that supports tools but cannot stream tool-call deltas
    # answers the request from chat() and the answer is emitted as chunks:
    # the client still gets a valid SSE stream, just not an incremental one.
    buffered_tools = bool(tool_kw) and not getattr(provider, "supports_tool_stream", False)

    async def _buffered_tool_stream(candidate):
        resp = await provider.chat(
            messages=candidate, model=model, max_tokens=_max_tokens(req),
            **_sampling_kwargs(req), **tool_kw,
        )
        if resp.text:
            yield resp.text
        if resp.tool_calls:
            yield {"tool_calls": [{**tc, "index": i} for i, tc in enumerate(resp.tool_calls)]}

    async def _gen():
        full_text = ""
        saw_tool_calls = False
        t0 = time.monotonic()
        cache_hit = False
        stream_failed = False
        transform_rejected = False
        yield _chunk({"role": "assistant"})
        try:
            cached = (
                _cache.get(user_content, session_id=session_id, context=cache_ctx)
                if settings.cache.enabled and user_content and not tool_kw
                else None
            )

            cache_hit = cached is not None

            if cached:
                full_text = cached.response
                yield _chunk({"content": full_text})
            else:
                candidates = [messages]
                if raw_messages is not None and raw_messages != messages:
                    candidates.append(raw_messages)

                last_error: Exception | None = None
                succeeded = False
                for attempt, candidate in enumerate(candidates):
                    last_error = None
                    try:
                        pieces = (
                            _buffered_tool_stream(candidate) if buffered_tools
                            else provider.chat_stream(
                                messages=candidate,
                                model=model,
                                max_tokens=_max_tokens(req),
                                **_sampling_kwargs(req), **tool_kw,
                            )
                        )
                        async for piece in pieces:
                            if isinstance(piece, dict):
                                # A tool-call delta (see OpenAIProvider.
                                # chat_stream): re-emitted in the same
                                # OpenAI chunk shape.
                                saw_tool_calls = True
                                yield _chunk({"tool_calls": piece["tool_calls"]})
                                continue
                            full_text += piece
                            yield _chunk({"content": piece})
                        succeeded = True
                        transform_rejected = attempt > 0
                        break
                    except Exception as e:
                        last_error = e
                        # Content already reached the client — an
                        # unrecoverable partial answer is out the door, so
                        # retrying now would duplicate or interleave
                        # output. This failure is final regardless of
                        # attempts left.
                        if full_text or saw_tool_calls:
                            break
                        if attempt == 0 and len(candidates) > 1:
                            correlation_id = uuid.uuid4().hex[:12]
                            logger.warning(
                                f"Provider rejected the transformed stream "
                                f"request [{correlation_id}] for session "
                                f"{session_id!r}; retrying with the "
                                f"untransformed messages: {type(e).__name__}"
                            )
                            continue
                        break

                if not succeeded:
                    raise last_error
                if transform_rejected:
                    logger.error(
                        f"TokenMizer's transformed stream request was "
                        f"rejected but the untransformed one succeeded for "
                        f"session {session_id!r} — a pipeline layer "
                        f"produced a request the provider will not accept. "
                        f"Original error: {last_error}"
                    )
                    _analytics.record_silent_failure("transform_rejected")
                    savings.clear()

        except ProviderError as e:
            # Mid-stream failure: SSE can't change the status code anymore —
            #emit an explicit error event instead of silently truncating.
            stream_failed = True
            yield "data: " + _json.dumps(
                {"error": {"message": str(e), "type": "provider_error"}}
            ) + "\n\n"
        except Exception as e:
            # Anything the provider layer didn't wrap in ProviderError
            # (transport errors, decode errors, bugs) would otherwise
            # escape the generator: the SSE stream dies with no
            # terminator and the bookkeeping below never runs.
            stream_failed = True
            correlation_id = uuid.uuid4().hex[:12]
            logger.error(f"Stream error [{correlation_id}]: {e}")
            yield "data: " + _json.dumps(
                {"error": {"message": f"Stream failed (ref: {correlation_id})",
                           "type": "internal_error"}}
            ) + "\n\n"
        yield _chunk(
            {}, finish="tool_calls" if saw_tool_calls else "stop",
            extra={"tokenmizer": {"fallback": {"transform_rejected": True}}}
            if transform_rejected else None,
        )
        yield "data: [DONE]\n\n"

        # Post-stream bookkeeping. When the transformed request was
        # rejected, what was actually SENT is raw_messages — mirrors
        # _call_provider()'s `sent_input_tokens = orig_input_tokens`.
        latency_ms = (time.monotonic() - t0) * 1000
        output_tokens = count_tokens(full_text, model)
        input_tokens = (
            orig_input_tokens if transform_rejected
            else count_messages_tokens(messages, model)
        )
        # Never cache a response that didn't finish. `full_text` after a
        # mid-stream failure holds however many tokens arrived before the
        # error — writing that to the cache would serve a silently
        # truncated answer to every future matching prompt, long after the
        # provider recovered. Re-writing a cache HIT is equally pointless.
        if (settings.cache.enabled and user_content and full_text
                and not stream_failed and not cache_hit and not saw_tool_calls):
            _cache.set(user_content, full_text, input_tokens=input_tokens,
                       output_tokens=output_tokens, session_id=session_id,
                       context=cache_ctx)
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
    # One exact-match substitution, never re-applied, so a map that points
    # at another of its own keys cannot loop. The original name is echoed
    # back below: a client that asked for one model and reads an answer
    # shaped like another should be able to see why from the response.
    mapped_from = None
    if model in settings.model_map:
        mapped_from, model = model, settings.model_map[model]
        logger.debug("model_map: %r -> %r (session %r)", mapped_from, model, session_id)
    savings: dict[str, int] = {}

    # Tool calling. `tools`/`tool_choice` are forwarded (see
    # providers/tools.py); the legacy `functions`/`function_call` pair is
    # not translated and, since ChatRequest uses extra="allow" so no
    # standard client ever gets a 422, its presence must at least not be
    # silent to whoever operates the proxy.
    _legacy = (req.model_extra or {}).keys() & {"functions", "function_call"}
    if _legacy:
        logger.warning(
            "Request for session %r used the deprecated %s fields — these "
            "are ignored. Send `tools`/`tool_choice` instead; the model "
            "will respond with no knowledge of the functions it was given.",
            session_id, sorted(_legacy),
        )
    if req.tools:
        try:
            provider_for_tools = _get_provider()
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))
        if not getattr(provider_for_tools, "supports_tools", False):
            raise HTTPException(
                status_code=501,
                detail=(f"Tool calling is not implemented for provider "
                        f"'{settings.provider}' (supported: anthropic, openai, "
                        f"deepseek, mistral, openrouter, grok, ollama). The "
                        f"request was refused rather than sent without its tools."),
            )

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
    #     (haiku/gpt-4o-mini/deepseek), a separate third party.
    #   - checkpoint storage (SQLite) and the graph DB itself
    # Redacting once here means every downstream path is safe by construction
    # instead of relying on each call site to remember to redact.
    raw_messages = [m.to_dict() for m in req.messages]
    raw_messages = redact_messages(raw_messages)
    # Per-dict copy, not raw_messages[:]. A shallow list copy shares every
    # dict, so Layer 2's terse-prompt injection — which prepends onto the
    # system message in place — rewrote raw_messages too. raw_messages is
    # the baseline: it feeds orig_input_tokens (the denominator of
    # tokens_saved and of every savings percentage reported to the client,
    # /api/stats and the dashboard), graph extraction, and checkpoint
    # storage. TokenMizer was counting its own injected prompt as part of
    # the request the user sent — overstating savings on every call, and
    # handing its own instructions to the extractor as session content.
    messages     = [dict(m) for m in raw_messages]
    user_query   = next(
        (m["content"] for m in reversed(raw_messages) if m.get("role") == "user"), ""
    )
    # The cache key is the final user turn. A request that ends on a tool
    # result (the agent loop's second half) or that declares tools is not
    # cacheable: the answer is a step in a plan, not a reply to a
    # question, and user_content="" disables both lookup and write.
    ends_on_user = bool(raw_messages) and raw_messages[-1].get("role") == "user"
    user_content = user_query if (ends_on_user and not req.tools) else ""

    # Layer 0-2: file intelligence, compression, terse injection
    messages = _apply_compression_layers(messages, settings, savings)

    orig_input_tokens = count_messages_tokens(raw_messages, model)

    # Layer 4: graph update + context injection (mutates messages)
    checkpoint_status: dict = {"attempted": False, "succeeded": False, "checkpoint_id": None}
    if settings.graph_checkpoint.enabled:
        graph    = await _get_graph_async(session_id)
        # Two separate protections, for two separate hazards.
        #
        # _session_in_use keeps a concurrent request's cache eviction from
        # force-persisting and detaching this instance out from under us,
        # which would silently drop everything added below.
        #
        # The session lock keeps this out of the background extractor's way.
        # _background_extract already runs under _get_session_lock and
        # mutates the same graph — extract_from_messages, prune() and
        # _persist() — but the foreground path took no lock at all, so turn
        # N's background extraction could interleave with turn N+1's
        # foreground mutation. Both mutators are synchronous, so this does
        # not corrupt the node dict outright; what it did allow is a
        # prune-to-200 and a persist-diff observing inconsistent
        # intermediate state. Rare, silent, and unreproducible for whoever
        # hits it. One lock, both paths.
        async with _get_session_lock(session_id):
            with _session_in_use(session_id):
                messages, checkpoint_status = await _update_graph(
                    session_id, graph, raw_messages, messages, model, savings,
                    user_query, principal,
                )

    # Measured AFTER _update_graph(), so it reflects both the reduction
    # from windowing and the addition from graph-context injection —
    # i.e. the real size of what is about to be sent.
    sent_input_tokens = count_messages_tokens(messages, model)

    # Streaming: true SSE passthrough (v0.3). Output-trimming is skipped in
    # stream mode (can't trim tokens that already left the building) — all
    # input-side layers (file intel, compression, graph context) still apply.
    if req.stream:
        return _stream_response(req, messages, model, user_content,
                                session_id, savings, orig_input_tokens,
                                raw_messages=raw_messages)

    # Layer 5: call provider (or return cache hit)
    outcome: dict = {}
    response_text, input_tokens_actual, output_tokens, latency_ms, cache_hit = await _call_provider(
        req, messages, model, user_content, session_id, savings,
        raw_messages=raw_messages, outcome=outcome,
    )
    # What the provider answered with, and — separately — whether the
    # transformed request was rejected on the way. Only the latter is
    # reported as `fallback`.
    tool_calls = outcome.pop("tool_calls", None) or []
    finish_reason = outcome.pop("finish_reason", None) or "stop"
    fallback = outcome
    if fallback:
        # The transformed request was rejected and the client's own
        # messages went through instead: what was sent is what they sent.
        sent_input_tokens = orig_input_tokens

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

    message: dict = {"role": "assistant", "content": response_text}
    if tool_calls:
        message["tool_calls"] = tool_calls
        # OpenAI sends null content for a pure tool-call turn; clients
        # (and their SDKs' pydantic models) accept "" but expect the key.
        message["content"] = response_text or None
        finish_reason = "tool_calls"
    elif finish_reason == "tool_calls":
        finish_reason = "stop"

    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   model,
        "session_id": session_id,
        "choices": [{
            "index":         0,
            "message":       message,
            "finish_reason": finish_reason,
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
            # Surfaced so a client can detect "my context wasn't saved"
            # here, rather than when a later resume returns nothing.
            "checkpoint": checkpoint_status,
            # Present only when TokenMizer's transformed request was
            # rejected and the turn went through untransformed. Zero
            # savings this turn, and a bug to report with `ref`.
            **({"fallback": fallback} if fallback else {}),
            # Present only when `model_map` substituted the model, so an
            # answer from a model the client never named is traceable to
            # the config rather than looking like a provider bug.
            **({"model_mapped_from": mapped_from} if mapped_from else {}),
        },
    }

# ── Health / Info ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Liveness AND the failure counters that were only visible in logs.

    `status` was the literal string "ok" whatever had happened, so a
    deployment whose checkpoints had been failing for a day, or whose
    graph database had been quarantined after corruption, reported
    healthy to every uptime monitor pointed at it. For a tool whose whole
    claim is "your context is safe", that is the one check that must not
    lie.

    "degraded" means the proxy is serving requests but something it was
    trusted to keep has failed: a write that did not land, a session
    whose stored graph could not be read, storage that is not durable, or
    memory displaced by corruption recovery. The detail says which, and
    every counter here is also available in /api/stats.
    """
    failures = _analytics.persist_failures
    sessions_load_failed = sorted(
        sid for sid, g in _graph_cache.items() if g._load_failed)
    sessions_no_durability = sorted(
        sid for sid, g in _graph_cache.items() if g._persistence_broken)
    sessions_data_loss = sorted(
        sid for sid, g in _graph_cache.items() if g._data_loss_detected)
    checkpoints_broken = bool(getattr(_checkpoint_mgr, "persistence_broken", False))
    checkpoints_data_loss = bool(getattr(_checkpoint_mgr, "data_loss_detected", False))

    degraded = bool(failures or sessions_load_failed or sessions_no_durability
                    or sessions_data_loss or checkpoints_broken or checkpoints_data_loss)
    return {
        "status": "degraded" if degraded else "ok",
        "timestamp": time.time(),
        "version": __version__,
        "sessions_in_memory": len(_graph_cache),
        # Non-zero means something was lost or not written. Each key names
        # the path that failed; see AnalyticsEngine.record_silent_failure.
        "persist_failures": failures,
        # Deliberate, bounded degradation — NOT part of `degraded`. A
        # proxy shedding the optional LLM extraction pass under load is
        # working as configured; calling that unhealthy would train an
        # operator to ignore the field that means something was lost.
        "shed": _analytics.shed,
        "sessions_with_unreadable_graph": sessions_load_failed,
        "sessions_without_durable_storage": sessions_no_durability,
        "sessions_with_data_loss": sessions_data_loss,
        "checkpoint_storage_broken": checkpoints_broken,
        "checkpoint_data_loss": checkpoints_data_loss,
    }


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
