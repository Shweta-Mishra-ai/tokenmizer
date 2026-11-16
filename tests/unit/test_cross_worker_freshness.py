"""A worker's cached copy of a session follows other workers' writes.

Each process of `uvicorn --workers N` caches its own GraphMemory per
session. Writes from several workers merge safely on disk, but a worker
kept answering from the copy it loaded first: in benchmarks/load with 4
workers, 23 of 48 sessions were answered with a decision that a later
turn, handled by another worker, had replaced.

A second GraphMemory instance on the same storage directory stands in for
the other worker; the file lock and the version counter are the same
either way (tests/unit/test_multiprocess.py covers real processes).
"""
from __future__ import annotations

import asyncio
import sqlite3

import pytest

from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType


def _g(tmp_path, sid="s"):
    return GraphMemory(sid, storage_dir=str(tmp_path))


def _labels(g):
    return {n.label for n in g._nodes.values()}


# ── the version counter ───────────────────────────────────────────────────

def test_a_write_by_another_instance_makes_this_one_stale(tmp_path):
    a, b = _g(tmp_path), _g(tmp_path)
    assert not a.is_stale() and not b.is_stale()

    b.add_node(NodeType.DECISION, "Use queue_7 as the partition key", NodeStatus.COMPLETED)
    assert b._persist()

    assert a.is_stale()
    assert not b.is_stale(), "an instance's own write must not mark it stale"


def test_a_merged_write_does_not_claim_to_be_current(tmp_path):
    """A writes after B without reading B's rows. The write merges on disk,
    but A's memory still lacks B's node, so A must still read as stale."""
    a, b = _g(tmp_path), _g(tmp_path)
    b.add_node(NodeType.DECISION, "Use queue_7 as the partition key", NodeStatus.COMPLETED)
    b._persist()

    a.add_node(NodeType.TASK, "Write the exporter docs", NodeStatus.PENDING)
    assert a._persist()
    assert a.is_stale()
    assert "Use queue_7 as the partition key" in _labels(_g(tmp_path))


def test_an_unchanged_session_is_not_stale(tmp_path):
    a = _g(tmp_path)
    a.add_node(NodeType.TASK, "Write the exporter docs", NodeStatus.PENDING)
    a._persist()
    assert not _g(tmp_path).is_stale()
    assert a._persist()            # nothing changed: no write, no new version
    assert not a.is_stale()


def test_a_database_from_before_the_counter_is_upgraded(tmp_path):
    db = tmp_path / "graph_memory.db"
    tmp_path.mkdir(exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE graph_meta (
        session_id TEXT PRIMARY KEY, processed_hashes TEXT NOT NULL DEFAULT '[]',
        schema_version INTEGER NOT NULL DEFAULT 2, updated_at REAL NOT NULL)""")
    conn.execute("INSERT INTO graph_meta VALUES ('s', '[]', 2, 1.0)")
    conn.commit()
    conn.close()

    a = _g(tmp_path)
    assert not a.is_stale()
    a.add_node(NodeType.TASK, "Write the exporter docs", NodeStatus.PENDING)
    assert a._persist()
    assert not a.is_stale()
    assert _g(tmp_path)._disk_version == 1


# ── the proxy's cache ─────────────────────────────────────────────────────

@pytest.fixture
def app_module(tmp_path, monkeypatch):
    import tokenmizer.api.app as app_module
    monkeypatch.setattr(app_module.settings.graph_checkpoint, "storage_dir", str(tmp_path))
    app_module._graph_cache.clear()
    yield app_module
    app_module._graph_cache.clear()


def test_the_cache_reloads_a_session_another_worker_wrote(tmp_path, app_module):
    async def run():
        mine = await app_module._get_graph_async("s")
        other = _g(tmp_path)      # the other worker
        other.add_node(NodeType.DECISION, "Use queue_9 as the partition key",
                       NodeStatus.COMPLETED)
        other._persist()
        return mine, await app_module._get_graph_async("s")

    before, after = asyncio.run(run())
    assert "Use queue_9 as the partition key" in _labels(after)
    assert app_module._graph_cache["s"] is after
    assert after is not before


def test_a_reload_keeps_this_workers_unsaved_changes(tmp_path, app_module):
    async def run():
        mine = await app_module._get_graph_async("s")
        mine.add_node(NodeType.TASK, "Write the exporter docs", NodeStatus.PENDING)
        other = _g(tmp_path)
        other.add_node(NodeType.DECISION, "Use queue_9 as the partition key",
                       NodeStatus.COMPLETED)
        other._persist()
        return await app_module._get_graph_async("s")

    labels = _labels(asyncio.run(run()))
    assert {"Write the exporter docs", "Use queue_9 as the partition key"} <= labels


def test_a_busy_session_is_returned_without_waiting(tmp_path, app_module):
    """Cross-session recall fetches other sessions' graphs while holding
    its own session's lock; waiting here for another session's lock could
    deadlock two such requests. A busy session is served as cached, and
    the proxy path re-checks it under its lock."""
    async def run():
        mine = await app_module._get_graph_async("s")
        other = _g(tmp_path)
        other.add_node(NodeType.TASK, "Write the exporter docs", NodeStatus.PENDING)
        other._persist()
        async with app_module._get_session_lock("s"):
            got = await asyncio.wait_for(app_module._get_graph_async("s"), timeout=2)
            assert got is mine
            refreshed = await app_module._refresh_graph("s", got)
        return refreshed

    assert "Write the exporter docs" in _labels(asyncio.run(run()))
