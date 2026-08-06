"""
Cross-process write safety.

The in-process asyncio.Lock does nothing across process boundaries, so
two uvicorn workers (or a worker and the CLI) holding the same session
each write what THEY believe the graph contains. Per-row storage stopped
that from being catastrophic; these tests cover what it did not fix on
its own — stale state overwriting fresh state — and the OS-level lock
that does.

Real subprocesses are used deliberately: threads share an interpreter and
would not exercise the file lock at all.
"""
from __future__ import annotations

import multiprocessing as mp

import pytest

from tokenmizer.graph_memory.filelock import LockUnavailable, session_lock
from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType


def _seed(storage, session="s", n=100):
    g = GraphMemory(session, storage_dir=str(storage))
    for i in range(n):
        g.add_node(NodeType.TASK, f"Implement feature number {i} in the API layer",
                   NodeStatus.COMPLETED)
    g._persist(force=True)
    return g


class TestStaleWriterCannotUndoAnother:
    def test_prune_survives_a_stale_writer(self, tmp_path):
        """Worker A prunes; worker B, still holding the pre-prune graph,
        must not reinstate every row A deleted."""
        _seed(tmp_path)

        a = GraphMemory("s", storage_dir=str(tmp_path))
        b = GraphMemory("s", storage_dir=str(tmp_path))
        assert len(a._nodes) == len(b._nodes) == 100

        a.prune(max_nodes=40)
        a._persist()

        b.add_node(NodeType.DECISION, "Use PostgreSQL for the user database",
                   NodeStatus.COMPLETED)
        b._persist()

        final = GraphMemory("s", storage_dir=str(tmp_path))
        assert len(final._nodes) <= 41, (
            f"a stale writer reinstated pruned rows: {len(final._nodes)} on disk"
        )
        assert any("PostgreSQL" in n.label for n in final._nodes.values()), (
            "the stale writer's genuinely new node was lost"
        )

    def test_new_work_is_never_dropped_by_reconciliation(self, tmp_path):
        """Adopting another process's deletions must not touch rows this
        instance created but has not yet persisted."""
        _seed(tmp_path, n=10)
        a = GraphMemory("s", storage_dir=str(tmp_path))
        b = GraphMemory("s", storage_dir=str(tmp_path))

        a.prune(max_nodes=2)
        a._persist()

        # Distinct decisions in distinct topic slots — labels differing
        # only by a digit would legitimately be deduplicated, which would
        # test the dedup rule rather than reconciliation.
        fresh = [
            "Use PostgreSQL for the user database",
            "Deploy the API on Kubernetes",
            "Use pytest as the test runner",
        ]
        for label in fresh:
            b.add_node(NodeType.DECISION, label, NodeStatus.COMPLETED)
        b._persist()

        labels = {n.label for n in GraphMemory("s", storage_dir=str(tmp_path))._nodes.values()}
        for label in fresh:
            assert label in labels, f"reconciliation dropped unpersisted new work: {label!r}"

    def test_other_process_additions_are_not_deleted(self, tmp_path):
        """Rows this instance never wrote belong to somebody else and
        must survive its writes."""
        a = GraphMemory("s-add", storage_dir=str(tmp_path))
        a.add_node(NodeType.TASK, "Worker A owns this task entirely", NodeStatus.COMPLETED)
        a._persist(force=True)

        b = GraphMemory("s-add", storage_dir=str(tmp_path))
        b.add_node(NodeType.TASK, "Worker B owns this other task", NodeStatus.COMPLETED)
        b._persist()

        labels = {n.label for n in GraphMemory("s-add", storage_dir=str(tmp_path))._nodes.values()}
        assert any("Worker A" in x for x in labels)
        assert any("Worker B" in x for x in labels)


class TestFileLock:
    def test_lock_is_exclusive_across_processes(self, tmp_path):
        holder_ready = mp.Event()
        release = mp.Event()

        # `_hold_lock` is module level, not a closure. Windows has no fork:
        # multiprocessing spawns a fresh interpreter and pickles the target,
        # and a function defined inside a test method cannot be pickled —
        # the child died with "Can't get local object".
        p = mp.Process(target=_hold_lock,
                       args=(str(tmp_path), holder_ready, release))
        p.start()
        try:
            assert holder_ready.wait(timeout=10), "helper never acquired the lock"
            with pytest.raises(LockUnavailable):
                with session_lock(tmp_path, "sess", timeout=0.5):
                    pass
        finally:
            release.set()
            p.join(timeout=10)

        # Once released it must be acquirable again.
        with session_lock(tmp_path, "sess", timeout=5) as locked:
            assert locked is True

    def test_different_sessions_do_not_contend(self, tmp_path):
        with session_lock(tmp_path, "session-one", timeout=1):
            with session_lock(tmp_path, "session-two", timeout=1) as locked:
                assert locked is True

    def test_hostile_session_ids_do_not_escape_the_lock_dir(self, tmp_path):
        """session_id is client-supplied, so it must not be able to steer
        the lock file outside .locks/ or collide with another session."""
        for sid in ("../../etc/passwd", "a/b/c", "x" * 500, "", "sess\x00id"):
            with session_lock(tmp_path, sid, timeout=1) as locked:
                assert locked is True
        created = list((tmp_path / ".locks").glob("*.lock"))
        assert len(created) == 5, "distinct session ids collided onto one lock file"
        for f in created:
            assert f.parent == tmp_path / ".locks"


def _hold_lock(storage_dir, ready, release):
    """Hold a session lock until told to let go.

    Module level so it can be pickled: Windows spawns a new interpreter
    for each process rather than forking, and pickles the target function
    into it.
    """
    from tokenmizer.graph_memory.filelock import session_lock
    with session_lock(storage_dir, "sess", timeout=5):
        ready.set()
        release.wait(timeout=10)


def _writer(storage_dir, worker_id, count, barrier):
    import logging
    logging.disable(logging.CRITICAL)
    from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType
    g = GraphMemory("shared", storage_dir=storage_dir)
    barrier.wait()          # start together, maximising interleaving
    for i in range(count):
        g.add_node(NodeType.TASK,
                   f"Worker {worker_id} implemented feature {i} in the API",
                   NodeStatus.COMPLETED)
        g._persist()


class TestConcurrentWriters:
    def test_four_processes_lose_nothing(self, tmp_path):
        workers, per_worker = 4, 15
        GraphMemory("shared", storage_dir=str(tmp_path))._persist(force=True)

        barrier = mp.Barrier(workers)
        procs = [
            mp.Process(target=_writer, args=(str(tmp_path), w, per_worker, barrier))
            for w in range(workers)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
            assert p.exitcode == 0, f"writer exited {p.exitcode}"

        labels = {n.label for n in GraphMemory("shared", storage_dir=str(tmp_path))._nodes.values()}
        for w in range(workers):
            got = sum(1 for x in labels if x.startswith(f"Worker {w} "))
            assert got == per_worker, f"worker {w} lost {per_worker - got} nodes"


class TestLockFileHousekeeping:
    """One lock file is created per session touched and never removed on
    release, so the directory would otherwise gain an inode per session
    forever. A test run committed 135 of them once."""

    def test_stale_locks_are_swept(self, tmp_path):
        import os
        import time

        from tokenmizer.graph_memory.filelock import lock_files_in, sweep_stale_locks

        for i in range(5):
            with session_lock(tmp_path, f"session-{i}", timeout=1):
                pass
        assert len(lock_files_in(tmp_path)) == 5

        old = time.time() - 40 * 86_400
        for f in lock_files_in(tmp_path)[:3]:
            os.utime(f, (old, old))

        assert sweep_stale_locks(tmp_path) == 3
        assert len(lock_files_in(tmp_path)) == 2, "recent locks must be kept"

    def test_a_held_lock_is_never_swept(self, tmp_path):
        import os
        import time

        from tokenmizer.graph_memory.filelock import lock_files_in, sweep_stale_locks

        with session_lock(tmp_path, "busy", timeout=1):
            for f in lock_files_in(tmp_path):
                old = time.time() - 90 * 86_400
                os.utime(f, (old, old))
            assert sweep_stale_locks(tmp_path) == 0, "swept a lock that was held"
            assert len(lock_files_in(tmp_path)) == 1
