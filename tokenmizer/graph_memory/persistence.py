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

Storage is per-row (schema v2); see init_db() and persist() for the
layout and for why change detection is derived rather than tracked.
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

# Storage schema version.
#   1 — one JSON blob per session in `graphs` (nodes_json/edges_json)
#   2 — one row per node and per edge (graph_nodes/graph_edges/graph_meta)
# Recorded per session in graph_meta so a mixed database is readable and
# migration can proceed one session at a time as each is next touched.
SCHEMA_VERSION = 2


def quarantine_db(db_path, reason: str) -> bool:
    """
    Move a corrupt DB file aside instead of deleting it, and report
    whether anything was actually displaced.

    This file is SHARED BY EVERY SESSION in a storage_dir — one
    `graph_memory.db` holds every session's nodes and edges. The
    previous behaviour on any DB-level error was
    `db_path.unlink(missing_ok=True)`, i.e. permanently destroying every
    session's graph because one session hit a bad read. Measured: three
    healthy sessions, one corrupt header, and the next unrelated session
    to connect wiped all three, with `_persistence_broken` still False
    so /api/graph/{id} reported healthy over an empty DB.

    Renaming keeps the bytes on disk. SQLite corruption is very often
    partial, so `.recover` against the quarantined file can usually get
    most of the graph back — impossible once it has been unlinked.
    """
    from pathlib import Path
    db_path = Path(db_path)
    if not db_path.exists():
        return False
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = db_path.with_name(f"{db_path.name}.corrupt-{stamp}")
    try:
        db_path.replace(target)
        logger.error(
            "Quarantined corrupt DB %s -> %s (%s). This file held EVERY "
            "session in this storage_dir; a fresh empty DB is being created. "
            "Recover with: sqlite3 %s '.recover' | sqlite3 %s",
            db_path, target, reason, target, db_path,
        )
        return True
    except Exception as move_err:
        logger.error("Could not quarantine corrupt DB %s: %s", db_path, move_err)
        return False


def safe_init_db(graph: "GraphMemory") -> None:
    """Initialize DB, quarantining a corrupt file if necessary."""
    try:
        graph._init_db()
    except Exception as init_err:
        if quarantine_db(graph._db_path, f"init failed: {init_err}"):
            # Data belonging to other sessions was just displaced. Say so
            # durably — this is the difference between "we started fresh"
            # and "we lost everyone's memory."
            graph._data_loss_detected = True
        try:
            graph._init_db()
        except Exception as e:
            logger.error(
                f"Cannot initialize DB after cleanup for {graph.session_id}: {e} "
                "— running with in-memory graph only (data won't persist)"
            )
            # Recorded on the instance, not just logged: this state
            # persists for the rest of the process's life and must be
            # queryable via stats().
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


def edge_key(edge: MemoryEdge) -> str:
    """Stable identity for an edge row.

    add_edge() already treats (source, target, type) as an edge's
    identity — it refuses to append a second edge with the same triple —
    so that triple is the natural primary key. Weight is data, not
    identity, and a weight change is an update to the same row.
    """
    etype = edge.type.value if hasattr(edge.type, "value") else str(edge.type)
    return f"{edge.source_id}|{edge.target_id}|{etype}"


def init_db(graph: "GraphMemory") -> None:
    conn = graph._db_connect()
    try:
        # ── Per-row storage (schema v2) ──────────────────────────────────
        #
        # Replaces the single nodes_json/edges_json blob per session (v1,
        # the `graphs` table below, kept for migration and rollback).
        #
        # The blob layout rewrote EVERY node and edge on every persist —
        # once per chat turn — even when one node changed, so write cost
        # was O(total graph) rather than O(changed). The 200-node
        # auto-prune cap exists to bound that cost, which is to say the
        # cap was a workaround for this schema rather than a memory
        # policy anyone wanted.
        #
        # It was also why a concurrent writer could destroy a whole
        # session: two processes holding the same session each wrote the
        # complete blob, so the later writer's version replaced the
        # earlier one wholesale, discarding every node the other had
        # added. With per-row writes, disjoint changes from two writers
        # now merge; only a genuine same-node conflict is last-writer-wins.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_nodes (
                session_id TEXT NOT NULL,
                node_id    TEXT NOT NULL,
                data_json  TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (session_id, node_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                session_id TEXT NOT NULL,
                edge_key   TEXT NOT NULL,
                data_json  TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (session_id, edge_key)
            )
        """)
        # Session-level state that isn't a node or an edge.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_meta (
                session_id       TEXT PRIMARY KEY,
                processed_hashes TEXT NOT NULL DEFAULT '[]',
                schema_version   INTEGER NOT NULL DEFAULT 2,
                updated_at       REAL NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_graph_nodes_session "
            "ON graph_nodes(session_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_graph_edges_session "
            "ON graph_edges(session_id)"
        )
        # v1 blob table. Still created (an older DB may not have it) and
        # still read by the one-time per-session migration in load(), but
        # never written again.
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


def _serialize_state(graph: "GraphMemory") -> tuple[dict, dict]:
    """Current nodes and edges as {key: canonical_json}.

    sort_keys makes the JSON canonical, so string equality against the
    previously written value is a reliable "did this row change?" test.
    """
    nodes = {
        nid: json.dumps(asdict(n), sort_keys=True, default=str)
        for nid, n in graph._nodes.items()
    }
    edges = {
        edge_key(e): json.dumps(asdict(e), sort_keys=True, default=str)
        for e in graph._edges
    }
    return nodes, edges


def persist(graph: "GraphMemory", force: bool = False) -> bool:
    """
    Persist the graph to SQLite, writing only what actually changed.

    Returns True if the graph's current state is confirmed on disk —
    either the write just succeeded, or nothing changed so the last
    successful write already covers it — and False if a write was
    attempted and failed.

    WHY CHANGE DETECTION IS DERIVED, NOT TRACKED
    --------------------------------------------
    The obvious way to write only changed rows is a dirty-set that every
    mutation site appends to. That would be a new way to lose data
    silently: node/edge state is mutated from at least six places
    (add_node's dedup, fuzzy-merge and supersede branches, add_edge,
    pruning's decay and eviction passes, and the /api/decision/invalidate
    endpoint mutating node.status directly), and any site that forgot to
    register would have its change quietly never reach disk. The existing
    `force=True` parameter exists precisely because one such site was
    already missed once.

    So the diff is derived from state instead: serialize the graph,
    compare each row against the exact string last written, write the
    differences. Serialization is O(nodes) CPU with no I/O, while the
    part that actually costs — SQLite writes, WAL appends, fsync — is
    O(changed). Nothing can be missed, because nothing has to be
    remembered.

    force=True rewrites every row regardless of the diff.
    """
    if not graph._dirty and not force:
        return True

    # Refuse to write over a row we failed to READ.
    #
    # An instance whose _load() failed (e.g. lock contention) holds an
    # empty or partial graph. Under the old blob layout that would
    # replace a good stored graph with nothing; per-row it would instead
    # delete every row it thinks was removed — same outcome. Either way a
    # transient, self-healing read failure must not become permanent
    # loss. Try the read once more; only proceed if it succeeds.
    if graph._load_failed:
        graph._load_failed = False
        graph._load()
        if graph._load_failed:
            logger.error(
                "Refusing to persist session %s: its stored state could "
                "not be read, and writing this instance\'s partial graph "
                "would overwrite it. Retrying on the next call.",
                graph.session_id,
            )
            return False

    nodes, edges = _serialize_state(graph)
    prev_nodes = {} if force else graph._persisted_nodes
    prev_edges = {} if force else graph._persisted_edges

    node_upserts = [(k, v) for k, v in nodes.items() if prev_nodes.get(k) != v]
    edge_upserts = [(k, v) for k, v in edges.items() if prev_edges.get(k) != v]
    # Deletions always diff against what was really written last, even
    # under force — pruning removes nodes and rebuilds the edge list, so
    # rows that no longer exist in memory have to go.
    node_deletes = [k for k in graph._persisted_nodes if k not in nodes]
    edge_deletes = [k for k in graph._persisted_edges if k not in edges]

    hashes_json = json.dumps(sorted(graph._processed_hashes))
    meta_changed = force or hashes_json != graph._persisted_hashes_json

    if not any((node_upserts, edge_upserts, node_deletes, edge_deletes, meta_changed)):
        graph._dirty = False
        return True

    now = time.time()
    sid = graph.session_id
    try:
        conn = graph._db_connect()
        try:
            # One transaction: a crash mid-write leaves the previous
            # committed state intact rather than a half-updated graph.
            with conn:
                if node_upserts:
                    conn.executemany(
                        "INSERT OR REPLACE INTO graph_nodes "
                        "(session_id, node_id, data_json, updated_at) VALUES (?,?,?,?)",
                        [(sid, k, v, now) for k, v in node_upserts],
                    )
                if edge_upserts:
                    conn.executemany(
                        "INSERT OR REPLACE INTO graph_edges "
                        "(session_id, edge_key, data_json, updated_at) VALUES (?,?,?,?)",
                        [(sid, k, v, now) for k, v in edge_upserts],
                    )
                if node_deletes:
                    conn.executemany(
                        "DELETE FROM graph_nodes WHERE session_id=? AND node_id=?",
                        [(sid, k) for k in node_deletes],
                    )
                if edge_deletes:
                    conn.executemany(
                        "DELETE FROM graph_edges WHERE session_id=? AND edge_key=?",
                        [(sid, k) for k in edge_deletes],
                    )
                conn.execute(
                    "INSERT OR REPLACE INTO graph_meta "
                    "(session_id, processed_hashes, schema_version, updated_at) "
                    "VALUES (?,?,?,?)",
                    (sid, hashes_json, SCHEMA_VERSION, now),
                )
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Graph persist failed for {graph.session_id}: {e}")
        # _persisted_* is left untouched, so the next call retries exactly
        # the same diff rather than assuming any of it landed.
        return False

    graph._persisted_nodes = nodes
    graph._persisted_edges = edges
    graph._persisted_hashes_json = hashes_json
    graph._dirty = False
    logger.debug(
        "Persisted %s: %d node upserts, %d edge upserts, %d/%d deletes "
        "(graph has %d nodes)",
        sid, len(node_upserts), len(edge_upserts),
        len(node_deletes), len(edge_deletes), len(nodes),
    )
    return True


def _hydrate_nodes(graph: "GraphMemory", nodes_data: list) -> None:
    """Rebuild MemoryNode objects from stored dicts.

    Enum restoration matters: asdict() writes NodeType/NodeStatus (both
    str-Enums) as plain strings, and because they compare equal to their
    values, a graph reloaded without conversion looks fine until
    something reads `.type.value` and crashes — which is what made every
    checkpoint of a reloaded session return HTTP 500 after a restart.
    """
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
        try:
            n = MemoryNode(**{k: v for k, v in nd.items() if k != "_evicted"})
        except TypeError as field_err:
            # A row written by a newer version with extra fields must not
            # take out the whole load.
            logger.warning(
                f"Skipping unreadable node row for {graph.session_id}: {field_err}"
            )
            continue
        graph._nodes[n.id] = n


def _hydrate_edges(graph: "GraphMemory", edges_data: list) -> None:
    for ed in edges_data:
        try:
            ed["type"] = EdgeType(ed["type"])
        except (ValueError, KeyError) as conv_err:
            logger.warning(
                f"Skipping edge with unknown type during load "
                f"for {graph.session_id}: {conv_err}"
            )
            continue
        try:
            graph._edges.append(MemoryEdge(**ed))
        except TypeError as field_err:
            logger.warning(
                f"Skipping unreadable edge row for {graph.session_id}: {field_err}"
            )


def _migrate_blob_session(graph: "GraphMemory", conn) -> bool:
    """One-time migration of one session from the v1 blob to per-row.

    Returns True if a v1 row existed and was loaded into memory. The
    rows are written to the new tables by the caller's normal persist
    path, so a failure here costs nothing — the blob is left untouched
    and the migration is simply retried next load.

    The v1 row is deliberately NOT deleted. Keeping it means downgrading
    to a pre-migration build still finds the data it expects (as of the
    moment of migration), which is what makes this rollback-safe. It
    does go stale from then on, so a downgrade loses changes made after
    the upgrade — that is the normal meaning of a rollback, and it is
    stated in the CHANGELOG rather than left for someone to discover.
    """
    row = conn.execute(
        "SELECT nodes_json, edges_json, processed_hashes FROM graphs "
        "WHERE session_id=?",
        (graph.session_id,),
    ).fetchone()
    if not row:
        return False

    nodes_data = json.loads(row[0])
    edges_data = json.loads(row[1])
    try:
        graph._processed_hashes = set(json.loads(row[2]))
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(
            f"processed_hashes corrupted for {graph.session_id}, resetting "
            f"(nodes/edges are unaffected): {e}"
        )
        graph._processed_hashes = set()

    _hydrate_nodes(graph, nodes_data)
    _hydrate_edges(graph, edges_data)
    logger.info(
        "Migrated session %s from v1 blob storage to per-row "
        "(%d nodes, %d edges); the v1 row is kept for rollback",
        graph.session_id, len(graph._nodes), len(graph._edges),
    )
    # Force the next persist to write every row into the new tables.
    graph._dirty = True
    graph._migrated_from_blob = True
    return True


def load(graph: "GraphMemory") -> None:
    try:
        conn = graph._db_connect()
        try:
            node_rows = conn.execute(
                "SELECT node_id, data_json FROM graph_nodes WHERE session_id=?",
                (graph.session_id,),
            ).fetchall()
            edge_rows = conn.execute(
                "SELECT edge_key, data_json FROM graph_edges WHERE session_id=?",
                (graph.session_id,),
            ).fetchall()
            meta_row = conn.execute(
                "SELECT processed_hashes FROM graph_meta WHERE session_id=?",
                (graph.session_id,),
            ).fetchone()

            if not node_rows and not edge_rows and meta_row is None:
                # Nothing in v2 for this session — it may predate the
                # migration. Falling back on "no v2 rows" rather than on a
                # global flag keeps this per-session, so a DB holding a
                # mix of migrated and unmigrated sessions converges one
                # session at a time as each is touched.
                if _migrate_blob_session(graph, conn):
                    return
                return
        finally:
            conn.close()

        # processed_hashes is parsed separately from nodes/edges: a
        # corrupt hash set costs only incremental-extraction dedup (some
        # messages get re-scanned, which add_node() dedupes anyway),
        # while nodes and edges are the irreplaceable part. Parsing them
        # together once meant a bad hash field discarded a perfectly good
        # graph.
        if meta_row is not None:
            try:
                graph._processed_hashes = set(json.loads(meta_row[0]))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    f"processed_hashes corrupted for {graph.session_id}, "
                    f"resetting (nodes/edges are unaffected): {e}"
                )
                graph._processed_hashes = set()

        # Per-row decoding: one unreadable row costs that row only,
        # instead of the whole session as a single bad blob did.
        good_nodes, good_edges = [], []
        for node_id, data_json in node_rows:
            try:
                good_nodes.append(json.loads(data_json))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    f"Skipping corrupt node row {node_id} for "
                    f"{graph.session_id} (other nodes unaffected): {e}"
                )
        for ekey, data_json in edge_rows:
            try:
                good_edges.append(json.loads(data_json))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    f"Skipping corrupt edge row {ekey} for "
                    f"{graph.session_id} (other edges unaffected): {e}"
                )

        _hydrate_nodes(graph, good_nodes)
        _hydrate_edges(graph, good_edges)

        # Record what is on disk so persist() can diff against it. Built
        # by re-serializing what we just hydrated, so it reflects exactly
        # what a persist() of the unchanged graph would produce; using
        # the raw stored strings would flag every row as changed whenever
        # a field default shifted.
        nodes, edges = _serialize_state(graph)
        graph._persisted_nodes = nodes
        graph._persisted_edges = edges
        graph._persisted_hashes_json = json.dumps(sorted(graph._processed_hashes))
        graph._dirty = False
    except sqlite3.OperationalError as e:
        # OperationalError is a SUBCLASS of DatabaseError and covers
        # entirely routine, transient conditions — above all "database is
        # locked" when a concurrent writer holds the lock past our 5s
        # busy timeout. The old handler caught DatabaseError and
        # OperationalError together and responded by deleting the shared
        # DB file, so ordinary write contention could destroy every
        # session's memory. Nothing is corrupt here and nothing may be
        # destroyed: fail this load, keep the file, let the next call
        # retry.
        logger.error(
            "Could not read graph for %s (%s: %s) — the database was NOT "
            "modified. In-memory graph is empty for this load; a retry "
            "should succeed once contention clears.",
            graph.session_id, type(e).__name__, e,
        )
        graph._nodes = {}
        graph._edges = []
        graph._processed_hashes = set()
        graph._load_failed = True
        # Do NOT clear _dirty here: this instance must not persist over a
        # row it failed to read, or a transient lock becomes real data loss.
        return
    except sqlite3.DatabaseError as e:
        logger.warning(f"Corrupted DB for {graph.session_id}: {e}")
        graph._nodes = {}
        graph._edges = []
        graph._processed_hashes = set()
        # Scope the blast radius: try dropping only THIS session's row
        # before touching a file that belongs to every other session too.
        try:
            conn = graph._db_connect()
            try:
                conn.execute("DELETE FROM graphs WHERE session_id=?", (graph.session_id,))
                conn.commit()
            finally:
                conn.close()
            logger.warning(
                "Recovered by dropping only session %s's row — other "
                "sessions in %s are untouched.",
                graph.session_id, graph._db_path,
            )
            graph._data_loss_detected = True
            return
        except Exception:
            pass  # row-scoped recovery failed — the file itself is bad
        try:
            if quarantine_db(graph._db_path, f"load failed: {e}"):
                graph._data_loss_detected = True
            graph._init_db()
        except Exception as reinit_err:
            logger.error(
                f"Graph DB reinit failed for {graph.session_id}: {reinit_err} "
                "— running with in-memory graph only (data won't persist)"
            )
            # Worst case: persistence is completely broken for this
            # session going forward. Surfaced via stats() so a health
            # check (or /api/graph/{session_id}) can detect "this session
            # has no durable memory" before a restart wipes everything.
            graph._persistence_broken = True
    except Exception as e:
        logger.warning(f"Graph load failed for {graph.session_id}: {e}")
