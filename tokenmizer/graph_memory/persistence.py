"""
Graph SQLite persistence — DB init, connect, load, and save.

Extracted from graph.py to keep that file focused on core memory logic
(node/edge CRUD, extraction application, query). Follows the same split
pattern already established for visualization.py and pruning.py:
module-level functions taking `graph: GraphMemory` as the first
argument, mutating its state directly, with GraphMemory keeping
one-line delegating methods so existing callers (`graph._persist()`,
`graph._load()`, `graph.get_transitions()`, etc. — used throughout
api/app.py, checkpoints/manager.py, semantic_cache/cache.py, and the
test suite) are unaffected.

Pure code motion — no logic changes. See graph.py's git history for
the reasoning behind each fix embedded in the comments below (TM-12
for the _persist() return-value fix, plus the partial-write recovery
and enum-restoration fixes in _load()).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict
from typing import TYPE_CHECKING

from tokenmizer.graph_memory.types import (
    DecisionTransition,
    EdgeType,
    MemoryEdge,
    MemoryNode,
    NodeStatus,
    NodeType,
)

if TYPE_CHECKING:
    from tokenmizer.graph_memory.graph import GraphMemory

logger = logging.getLogger(__name__)


def safe_init_db(graph: "GraphMemory") -> None:
    """Initialize DB, deleting corrupt file if necessary."""
    try:
        graph._init_db()
    except Exception:
        logger.warning(f"DB corrupt or unreadable — recreating: {graph._db_path}")
        try:
            graph._db_path.unlink(missing_ok=True)
        except Exception as del_err:
            logger.error(f"Could not delete corrupt graph DB: {del_err}")
        try:
            graph._init_db()
        except Exception as e:
            logger.error(
                f"Cannot initialize DB after cleanup for {graph.session_id}: {e} "
                "— running with in-memory graph only (data won't persist)"
            )
            # FIXED: previously this was a dead end — logged once at
            # startup and then silently true for the rest of the
            # process's life with no way to query it. See _load()'s
            # matching reinit-failure path for the same fix.
            graph._persistence_broken = True


def db_connect(graph: "GraphMemory") -> "sqlite3.Connection":
    """
    Open a SQLite connection with safe concurrent settings:
    - WAL journal mode: readers don't block writers, writers don't block readers
    - 5s timeout: prevents instant failure when another process holds a write lock
    - check_same_thread=False: safe because we serialize via asyncio session locks
    """
    conn = sqlite3.connect(str(graph._db_path), timeout=5.0, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")  # WAL + NORMAL = safe + fast
    except Exception:
        # Close before re-raising: a failed PRAGMA (corrupt DB file)
        # would otherwise leak the handle, and an open handle blocks
        # the delete-and-recreate recovery path on Windows.
        conn.close()
        raise
    return conn


def init_db(graph: "GraphMemory") -> None:
    conn = graph._db_connect()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS graphs (
                session_id TEXT PRIMARY KEY,
                nodes_json TEXT NOT NULL,
                edges_json TEXT NOT NULL,
                processed_hashes TEXT NOT NULL DEFAULT '[]',
                updated_at REAL NOT NULL
            )
        """)
        # Separate table for decision transitions — full causal story
        # Kept separate so it survives graph pruning and is queryable independently
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decision_transitions (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                from_decision_id TEXT NOT NULL,
                to_decision_id TEXT NOT NULL,
                from_label TEXT NOT NULL,
                to_label TEXT NOT NULL,
                trigger TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                evidence TEXT NOT NULL DEFAULT '',
                confidence_delta REAL NOT NULL DEFAULT 0.0,
                timestamp REAL NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


def persist_transition(graph: "GraphMemory", t: DecisionTransition) -> None:
    """Persist a single DecisionTransition to its own SQLite table."""
    try:
        conn = graph._db_connect()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO decision_transitions
                   (id, session_id, from_decision_id, to_decision_id,
                    from_label, to_label, trigger, reason, evidence,
                    confidence_delta, timestamp)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (t.id, t.session_id, t.from_decision_id, t.to_decision_id,
                 t.from_label, t.to_label, t.trigger, t.reason,
                 t.evidence, t.confidence_delta, t.timestamp),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Transition persist failed: {e}")


def get_transitions(graph: "GraphMemory") -> list[DecisionTransition]:
    """Return all decision transitions for this session, newest first."""
    return sorted(graph._transitions, key=lambda t: t.timestamp, reverse=True)


def load_transitions(graph: "GraphMemory") -> None:
    """Load transitions from SQLite into memory."""
    try:
        conn = graph._db_connect()
        try:
            rows = conn.execute(
                "SELECT id,session_id,from_decision_id,to_decision_id,"
                "from_label,to_label,trigger,reason,evidence,"
                "confidence_delta,timestamp "
                "FROM decision_transitions WHERE session_id=?",
                (graph.session_id,),
            ).fetchall()
            graph._transitions = [
                DecisionTransition(
                    id=r[0], session_id=r[1],
                    from_decision_id=r[2], to_decision_id=r[3],
                    from_label=r[4], to_label=r[5],
                    trigger=r[6], reason=r[7], evidence=r[8],
                    confidence_delta=r[9], timestamp=r[10],
                )
                for r in rows
            ]
        finally:
            conn.close()
    except Exception as e:
        # Distinguish the benign first-run case (table not created yet,
        # debug) from real failures such as corruption or schema
        # mismatch, which must be visible at default log levels.
        if "no such table" in str(e).lower():
            logger.debug(f"Transition load skipped (first run, table not "
                         f"created yet): {e}")
        else:
            logger.warning(f"Transition load FAILED for session "
                           f"{graph.session_id} — decision supersession "
                           f"history unavailable this session: "
                           f"{type(e).__name__}: {e}")
        graph._transitions = []


def persist(graph: "GraphMemory", force: bool = False) -> bool:
    """
    Persist the full graph (all nodes + edges) as JSON to SQLite.

    Returns True if the graph's current state is confirmed safely on
    disk — either the write just succeeded, or nothing was dirty so
    the last successful write already covers it — and False if a
    write was attempted and failed.

    FIXED (TM-12): this used to return None unconditionally, catching
    and logging its own exceptions internally without ever re-raising.
    api/app.py's graph-cache eviction path (_graph_cache_touch) wraps
    `evicted_graph._persist()` in a try/except with a retry-once loop
    specifically to handle transient failures — but since this method
    never raised, that except clause could never fire, `persisted =
    True` was set unconditionally on the very first call, and the
    retry (plus the record_silent_failure metric it guards) was dead
    code. A detailed comment there described that retry/alerting
    behavior as implemented; it never executed once. Returning bool
    lets the caller check the actual outcome instead of relying on an
    exception that was never going to arrive.

    KNOWN SCALING LIMITATION (documented, not silently shipped as if
    it were fine): this rewrites EVERY node and edge as JSON on every
    call, even when only 1-2 nodes actually changed. Cost is O(total
    node count) per persist, and persist is called once per chat turn
    in extract_from_messages(). The existing 200-node auto-prune cap
    (see prune()) is itself evidence this was already a known
    bottleneck — it caps the damage rather than fixing the cause.

    Why this isn't rewritten to a proper per-node table in this pass:
    that's a real schema migration (one row per node/edge instead of
    one JSON blob per session), and shipping a migration without being
    able to run it against real persisted data in this environment
    (no app runtime available here — see repo's TESTING.md) is exactly
    the kind of "looks fixed, silently corrupts production data" risk
    this audit is supposed to eliminate, not introduce. Tracked as a
    documented follow-up: migrate `graphs.nodes_json` blob storage to
    a `graph_nodes(session_id, node_id, data_json, updated_at)` table
    with per-node INSERT OR REPLACE, validated against a copy of real
    checkpoint data before rollout.

    What IS fixed here: a dirty flag so we skip the rewrite entirely
    when nothing changed since the last successful persist (e.g. a
    message produced zero new/updated nodes — common for short
    acknowledgement turns). This doesn't fix the O(n) cost when a
    write IS needed, but it does eliminate redundant full-rewrites,
    which in practice is a meaningful fraction of calls.

    force=True bypasses the dirty check. Required for callers that
    mutate node/edge state directly without going through add_node()
    (which sets the dirty flag) — e.g. the /api/decision/invalidate
    endpoint flips `node.status` directly, then must force a write or
    the change would silently never be saved.
    """
    if not graph._dirty and not force:
        return True
    try:
        conn = graph._db_connect()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO graphs
                   (session_id, nodes_json, edges_json, processed_hashes, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    graph.session_id,
                    json.dumps([asdict(n) for n in graph._nodes.values()]),
                    json.dumps([asdict(e) for e in graph._edges]),
                    json.dumps(list(graph._processed_hashes)),
                    time.time(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        graph._dirty = False  # only clear on confirmed success
        return True
    except Exception as e:
        logger.error(f"Graph persist failed for {graph.session_id}: {e}")
        # _dirty stays True — next call (even non-forced) will retry
        # the full write rather than silently giving up on it forever.
        return False


def load(graph: "GraphMemory") -> None:
    try:
        conn = graph._db_connect()
        try:
            row = conn.execute(
                "SELECT nodes_json, edges_json, processed_hashes FROM graphs WHERE session_id=?",
                (graph.session_id,),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return
        nodes_data = json.loads(row[0])
        edges_data = json.loads(row[1])

        # FIXED — real bug, found while writing a proper (non-vacuous) test
        # for tests/chaos/test_recovery.py::test_partial_write_recovery.
        # processed_hashes used to be parsed inline with nodes/edges, all
        # inside the same try block. If processed_hashes was corrupted
        # (e.g. a partial/interrupted write), json.loads() on it raised
        # BEFORE the node-population loop below ever ran — so a session
        # with perfectly valid nodes_json still lost every node on reload,
        # just because the unrelated hashes field was bad. That directly
        # contradicts this method's whole purpose (recover what's good).
        # Isolating this parse means a corrupt hash set only costs you
        # incremental-extraction dedup (some messages get re-processed —
        # harmless, add_node() already dedupes), not your entire graph.
        try:
            graph._processed_hashes = set(json.loads(row[2]))
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(
                f"processed_hashes corrupted for {graph.session_id}, "
                f"resetting (nodes/edges are unaffected): {e}"
            )
            graph._processed_hashes = set()

        # CRITICAL: restore enum types on load. asdict() serializes
        # NodeType/NodeStatus (str-Enums) to plain strings; without
        # converting back, every reloaded node has type/status as `str`.
        # Because they're str-Enums, equality checks still pass — which
        # HID this bug — but any `.value` access crashes ('str' object
        # has no attribute 'value'). Concretely: after a server restart,
        # every checkpoint of a reloaded session returned HTTP 500.
        # Found by the MCP e2e check, not by unit tests, because unit
        # tests reloaded graphs but never then called `.type.value`.
        for nd in nodes_data:
            nd.pop("_evicted", None)
            try:
                nd["type"] = NodeType(nd["type"])
                nd["status"] = NodeStatus(nd["status"])
            except (ValueError, KeyError) as conv_err:
                logger.warning(
                    f"Skipping node with unknown type/status during load "
                    f"for {graph.session_id}: {conv_err} — {nd.get('id')}"
                )
                continue
            n = MemoryNode(**{k: v for k, v in nd.items() if k != "_evicted"})
            graph._nodes[n.id] = n
        for ed in edges_data:
            try:
                ed["type"] = EdgeType(ed["type"])
            except (ValueError, KeyError) as conv_err:
                logger.warning(
                    f"Skipping edge with unknown type during load "
                    f"for {graph.session_id}: {conv_err}"
                )
                continue
            graph._edges.append(MemoryEdge(**ed))
    except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
        logger.warning(f"Corrupted DB for {graph.session_id} — starting fresh: {e}")
        graph._nodes = {}
        graph._edges = []
        graph._processed_hashes = set()
        # Re-initialize the DB file
        try:
            graph._db_path.unlink(missing_ok=True)
            graph._init_db()
        except Exception as reinit_err:
            logger.error(
                f"Graph DB reinit failed for {graph.session_id}: {reinit_err} "
                "— running with in-memory graph only (data won't persist)"
            )
            # FIXED: this is the worst-case path — persistence is
            # completely broken for this session going forward, but
            # previously the only trace of that fact was a log line.
            # Surfacing it via stats() means a health-check script (or
            # a human looking at /api/graph/{session_id}) can detect
            # "this session has no durable memory" instead of finding
            # out only after a restart wipes everything.
            graph._persistence_broken = True
    except Exception as e:
        logger.warning(f"Graph load failed for {graph.session_id}: {e}")
