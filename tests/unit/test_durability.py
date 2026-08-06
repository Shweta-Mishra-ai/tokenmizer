"""
Mid-session durability: nothing already accepted may be lost.

Covers the three windows in which a session's graph memory could
previously disappear even though nothing was "broken":

  1. Ordinary shutdown (SIGTERM from docker stop / a k8s rollout).
     The lifespan hook logged a line and exited; every dirty graph in
     _graph_cache was dropped unwritten and in-flight background
     extraction was killed mid-run.
  2. Cache eviction while the database is refusing writes. The evicted
     graph was popped from memory whether or not the flush succeeded,
     so a failing write meant permanent loss.
  3. Eviction of a session a request is actively using. The "is it in
     use?" test read a lock that only the background task ever took, so
     it was inert for request traffic.

Plus the storage-layer rule that makes the rest safe: a transient read
failure must never be converted into a destructive write.
"""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

import tokenmizer.api.app as app_module
from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_module.settings.graph_checkpoint, "storage_dir", str(tmp_path)
    )
    app_module._graph_cache.clear()
    app_module._session_inflight.clear()
    yield str(tmp_path)
    app_module._graph_cache.clear()
    app_module._session_inflight.clear()


def _graph_with_node(session_id, storage_dir, label="Use PostgreSQL for the user database"):
    g = GraphMemory(session_id, storage_dir=storage_dir)
    g.add_node(NodeType.DECISION, label, NodeStatus.COMPLETED)
    return g


class TestShutdownFlush:
    def test_shutdown_persists_dirty_graphs(self, storage):
        """A clean shutdown must write every cached graph to disk."""
        g = _graph_with_node("s-shutdown", storage)
        g._dirty = True
        app_module._graph_cache["s-shutdown"] = g

        with TestClient(app_module.app) as client:   # __exit__ runs the shutdown hook
            client.get("/health")

        reloaded = GraphMemory("s-shutdown", storage_dir=storage)
        assert len(reloaded._nodes) == 1, "graph was lost on shutdown"

    @pytest.mark.asyncio
    async def test_flush_reports_failures(self, storage, monkeypatch):
        """A flush that cannot write must be counted, not swallowed."""
        g = _graph_with_node("s-failing", storage)
        monkeypatch.setattr(g, "_persist", lambda force=False: False)
        app_module._graph_cache["s-failing"] = g

        flushed, failed = await app_module._flush_all_graphs("test")
        assert (flushed, failed) == (0, 1)


class TestEvictionNeverLosesData:
    def test_failed_persist_keeps_graph_in_memory(self, storage, monkeypatch):
        """If the graph cannot be written, it must stay in memory.

        Dropping it would destroy every node added since the last
        successful save — the exact outcome this product exists to
        prevent. Running over the cache cap is the cheaper failure.
        """
        victim = _graph_with_node("s-victim", storage)
        monkeypatch.setattr(victim, "_persist", lambda force=False: False)
        app_module._graph_cache["s-victim"] = victim

        for i in range(app_module._GRAPH_CACHE_MAX + 2):
            sid = f"filler-{i}"
            app_module._graph_cache[sid] = GraphMemory(sid, storage_dir=storage)
            app_module._graph_cache_touch(sid)

        assert "s-victim" in app_module._graph_cache, "unpersisted graph was evicted"
        assert len(app_module._graph_cache["s-victim"]._nodes) == 1

    def test_in_flight_session_is_not_evictable(self, storage):
        """A session a request is using must not be chosen for eviction."""
        app_module._graph_cache["busy"] = GraphMemory("busy", storage_dir=storage)

        with app_module._session_in_use("busy"):
            assert app_module._find_evictable_graph_id() is None
        assert app_module._find_evictable_graph_id() == "busy"

    def test_in_use_marker_is_reentrant(self, storage):
        """Concurrent requests on one session must not clear each other."""
        app_module._graph_cache["busy"] = GraphMemory("busy", storage_dir=storage)
        with app_module._session_in_use("busy"):
            with app_module._session_in_use("busy"):
                pass
            assert app_module._find_evictable_graph_id() is None, (
                "inner request finishing released the outer request's claim"
            )


class TestTransientFailureIsNotDestructive:
    def test_lock_error_does_not_destroy_the_database(self, storage):
        """'database is locked' is routine contention, not corruption.

        sqlite3.OperationalError subclasses DatabaseError, so the old
        recovery path treated ordinary write contention as a corrupt
        file and deleted the shared DB — losing every session's memory.
        """
        _graph_with_node("s-real", storage)._persist(force=True)

        g = GraphMemory("s-real", storage_dir=storage)
        assert len(g._nodes) == 1

        real_connect = g._db_connect

        def boom():
            raise sqlite3.OperationalError("database is locked")

        g._db_connect = boom
        g._nodes.clear()
        g._load()
        g._db_connect = real_connect

        assert g._load_failed is True
        # The stored row must be untouched and still readable.
        fresh = GraphMemory("s-real", storage_dir=storage)
        assert len(fresh._nodes) == 1, "a transient lock destroyed stored data"

    def test_persist_refuses_to_overwrite_an_unreadable_row(self, storage, monkeypatch):
        """A graph whose load failed must not write its empty state back.

        persist() writes the whole session as one blob, so persisting
        from an instance that failed to read would replace a good graph
        with nothing.
        """
        _graph_with_node("s-guard", storage)._persist(force=True)

        g = GraphMemory("s-guard", storage_dir=storage)
        g._nodes.clear()
        g._load_failed = True

        def _still_failing():
            # persist() retries the read before giving up; this stands in
            # for a lock that has not cleared yet.
            g._load_failed = True

        monkeypatch.setattr(g, "_load", _still_failing)

        assert g._persist(force=True) is False
        assert len(GraphMemory("s-guard", storage_dir=storage)._nodes) == 1


class TestCorruptionRecoveryIsScoped:
    def test_corrupt_db_is_quarantined_not_deleted(self, storage):
        """The DB file holds EVERY session — it must never be unlinked."""
        import glob
        import os

        for sid in ("alice", "bob"):
            _graph_with_node(sid, storage, f"{sid} picked PostgreSQL")._persist(force=True)

        db = os.path.join(storage, "graph_memory.db")
        with open(db, "r+b") as f:
            f.seek(0)
            f.write(b"\x00" * 64)

        dave = GraphMemory("dave", storage_dir=storage)

        quarantined = glob.glob(db + ".corrupt-*")
        assert quarantined, "corrupt DB was destroyed instead of quarantined"
        assert os.path.getsize(quarantined[0]) > 0
        assert dave._data_loss_detected is True
        assert dave.stats()["data_loss_detected"] is True, (
            "data loss must be queryable, not just logged"
        )


class TestPerRowPersistence:
    """
    Schema v2. The v1 layout stored one JSON blob per session, so every
    persist rewrote the entire graph (once per chat turn), and two
    writers holding the same session each wrote the complete blob —
    whoever saved last silently discarded everything the other added.
    """

    def _big_graph(self, storage, n=120):
        g = GraphMemory("s-rows", storage_dir=storage)
        for i in range(n):
            g.add_node(NodeType.TASK, f"Implement feature number {i} in the API layer",
                       NodeStatus.COMPLETED)
        g._persist(force=True)
        return g

    def _stamps(self, storage, session="s-rows"):
        conn = sqlite3.connect(str(storage) + "/graph_memory.db")
        try:
            return [r[0] for r in conn.execute(
                "SELECT updated_at FROM graph_nodes WHERE session_id=?", (session,))]
        finally:
            conn.close()

    def test_one_change_writes_one_row(self, storage):
        import time
        g = self._big_graph(storage)
        before = max(self._stamps(storage))
        time.sleep(0.01)

        g.add_node(NodeType.DECISION, "Use PostgreSQL for the user database",
                   NodeStatus.COMPLETED)
        g._persist()

        rewritten = sum(1 for s in self._stamps(storage) if s > before)
        assert rewritten == 1, (
            f"adding one node rewrote {rewritten} rows; the v1 blob rewrote "
            f"all {len(g._nodes)} on every single persist"
        )

    def test_noop_persist_writes_nothing(self, storage):
        import time
        g = self._big_graph(storage)
        time.sleep(0.01)
        mark = max(self._stamps(storage))
        g._persist()
        assert sum(1 for s in self._stamps(storage) if s > mark) == 0

    def test_deletions_propagate(self, storage):
        """prune() deletes nodes outright — the rows must go too, or the
        graph would resurrect pruned nodes on the next load."""
        g = self._big_graph(storage)
        g.prune(max_nodes=50)
        g._persist()

        conn = sqlite3.connect(str(storage) + "/graph_memory.db")
        try:
            rows = conn.execute(
                "SELECT COUNT(*) FROM graph_nodes WHERE session_id='s-rows'"
            ).fetchone()[0]
        finally:
            conn.close()
        assert rows == len(g._nodes)
        assert len(GraphMemory("s-rows", storage_dir=storage)._nodes) == len(g._nodes)

    def test_changes_are_detected_without_dirty_bookkeeping(self, storage):
        """A direct field mutation that bypasses add_node() must still be
        written. /api/decision/invalidate mutates node.status in place;
        a dirty-set that mutation sites had to register with would miss
        exactly this, which is why the diff is derived from state."""
        g = GraphMemory("s-rows", storage_dir=storage)
        nid = g.add_node(NodeType.DECISION, "Use MongoDB for the user database",
                         NodeStatus.COMPLETED)
        g._persist(force=True)

        g._nodes[nid].status = NodeStatus.INVALIDATED   # no dirty flag set
        g._persist(force=True)

        assert GraphMemory("s-rows", storage_dir=storage)._nodes[nid].status \
            == NodeStatus.INVALIDATED

    def test_concurrent_writers_merge_instead_of_clobbering(self, storage):
        a = GraphMemory("shared", storage_dir=storage)
        b = GraphMemory("shared", storage_dir=storage)

        a.add_node(NodeType.TASK, "Worker A implemented the billing endpoint",
                   NodeStatus.COMPLETED)
        b.add_node(NodeType.TASK, "Worker B implemented the auth endpoint",
                   NodeStatus.COMPLETED)
        a._persist()
        b._persist()          # under v1 this replaced A's entire blob

        labels = {n.label for n in GraphMemory("shared", storage_dir=storage)._nodes.values()}
        assert any("Worker A" in x for x in labels), "second writer erased the first"
        assert any("Worker B" in x for x in labels)

    def test_v1_blob_is_migrated_and_kept_for_rollback(self, storage):
        """Existing deployments hold v1 data; opening a session must
        migrate it losslessly, and must not destroy the v1 row (that is
        what makes downgrading possible)."""
        import json
        import time
        from dataclasses import asdict

        seed = GraphMemory("legacy", storage_dir=storage)
        for i in range(10):
            seed.add_node(NodeType.TASK, f"Legacy task number {i} in the pipeline",
                          NodeStatus.COMPLETED)
        seed.add_node(NodeType.DECISION, "Use PostgreSQL for the user database",
                      NodeStatus.COMPLETED)
        expected = sorted(n.label for n in seed._nodes.values())

        # Rewrite this session as a pure v1 database.
        conn = sqlite3.connect(str(storage) + "/graph_memory.db")
        try:
            conn.execute("DELETE FROM graph_nodes WHERE session_id='legacy'")
            conn.execute("DELETE FROM graph_edges WHERE session_id='legacy'")
            conn.execute("DELETE FROM graph_meta  WHERE session_id='legacy'")
            conn.execute(
                "INSERT OR REPLACE INTO graphs VALUES (?,?,?,?,?)",
                ("legacy",
                 json.dumps([asdict(n) for n in seed._nodes.values()]),
                 json.dumps([asdict(e) for e in seed._edges]),
                 json.dumps(list(seed._processed_hashes)), time.time()),
            )
            conn.commit()
        finally:
            conn.close()

        migrated = GraphMemory("legacy", storage_dir=storage)
        assert migrated._migrated_from_blob is True
        assert sorted(n.label for n in migrated._nodes.values()) == expected
        migrated._persist()

        conn = sqlite3.connect(str(storage) + "/graph_memory.db")
        try:
            rows = conn.execute(
                "SELECT COUNT(*) FROM graph_nodes WHERE session_id='legacy'"
            ).fetchone()[0]
            blob_kept = conn.execute(
                "SELECT COUNT(*) FROM graphs WHERE session_id='legacy'"
            ).fetchone()[0]
        finally:
            conn.close()
        assert rows == len(expected)
        assert blob_kept == 1, "the v1 row is the rollback path — do not delete it"

        # And the round-trip through v2 is lossless.
        assert sorted(n.label for n in
                      GraphMemory("legacy", storage_dir=storage)._nodes.values()) == expected

    def test_one_corrupt_row_does_not_lose_the_session(self, storage):
        """The v1 blob was all-or-nothing: one bad byte cost every node.
        Per-row, damage is contained to the damaged row."""
        g = GraphMemory("s-rows", storage_dir=storage)
        for i in range(5):
            g.add_node(NodeType.TASK, f"Implement feature number {i} in the API layer",
                       NodeStatus.COMPLETED)
        g._persist(force=True)

        conn = sqlite3.connect(str(storage) + "/graph_memory.db")
        try:
            victim = conn.execute(
                "SELECT node_id FROM graph_nodes WHERE session_id='s-rows' LIMIT 1"
            ).fetchone()[0]
            conn.execute(
                "UPDATE graph_nodes SET data_json='{not valid json' "
                "WHERE session_id='s-rows' AND node_id=?", (victim,))
            conn.commit()
        finally:
            conn.close()

        reloaded = GraphMemory("s-rows", storage_dir=storage)
        assert len(reloaded._nodes) == 4, "one corrupt row should cost one node"
        assert victim not in reloaded._nodes
