"""
Analytics must survive a restart.

Graph state has WAL-mode SQLite, a periodic flusher, corruption quarantine
and a shutdown drain. Analytics had none of it: records lived in a Python
list and nothing was ever written to disk, so daily(), weekly() and monthly()
reset to zero on every restart, redeploy or container recycle — and `stats`
is one of the first things a new user tries. A weekly figure that cannot
survive a week is not a weekly figure.
"""
from __future__ import annotations

import sqlite3

from tokenmizer.analytics.engine import AnalyticsEngine


def _record(engine, **overrides):
    kwargs = dict(
        session_id="s1", provider="anthropic", model="claude-sonnet-4-6",
        input_tokens_original=1000, input_tokens_sent=600, output_tokens=200,
        tokens_saved=400, latency_ms=120.0, cache_hit=False,
        layer_savings={"compression": 250, "windowing": 150},
    )
    kwargs.update(overrides)
    engine.record(**kwargs)


def test_records_survive_a_restart(tmp_path):
    first = AnalyticsEngine(storage_dir=str(tmp_path))
    for _ in range(3):
        _record(first)
    assert first.flush()
    before = first.daily

    restarted = AnalyticsEngine(storage_dir=str(tmp_path))
    after = restarted.daily

    assert after.requests == before.requests == 3
    assert after.tokens_saved == before.tokens_saved == 1200


def test_layer_breakdown_survives_a_restart(tmp_path):
    first = AnalyticsEngine(storage_dir=str(tmp_path))
    _record(first)
    first.flush()

    restarted = AnalyticsEngine(storage_dir=str(tmp_path))
    assert restarted.layer_breakdown()["compression"] == 250


def test_lifetime_counters_survive_record_trimming(tmp_path):
    """The engine's own rule: a total that silently decreases as old records
    age out is worse than no total at all. That has to hold across a restart
    too, where only the retained window is reloaded."""
    first = AnalyticsEngine(storage_dir=str(tmp_path))
    for _ in range(5):
        _record(first)
    first.flush()

    restarted = AnalyticsEngine(storage_dir=str(tmp_path))
    assert restarted._total_requests == 5
    assert restarted._provider_totals["anthropic"] == 5


def test_records_are_buffered_not_written_per_request(tmp_path):
    """record() is called synchronously from the chat handler; an fsync per
    request would put disk latency on the hot path."""
    engine = AnalyticsEngine(storage_dir=str(tmp_path))
    _record(engine)

    with sqlite3.connect(str(tmp_path / "analytics.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0

    engine.flush()
    with sqlite3.connect(str(tmp_path / "analytics.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 1


def test_old_rows_are_deleted_on_flush(tmp_path):
    engine = AnalyticsEngine(storage_dir=str(tmp_path))
    _record(engine)
    engine.flush()

    with sqlite3.connect(str(tmp_path / "analytics.db")) as conn:
        conn.execute("UPDATE requests SET timestamp = 1.0")

    _record(engine)
    engine.flush()

    with sqlite3.connect(str(tmp_path / "analytics.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 1


def test_a_failed_flush_keeps_the_records_for_the_next_attempt(tmp_path, monkeypatch):
    """Dropping the buffer on a transient error would lose the window
    silently — the failure mode this whole change exists to remove."""
    engine = AnalyticsEngine(storage_dir=str(tmp_path))
    _record(engine)

    def boom():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(engine, "_connect", boom)
    assert engine.flush() is False
    assert len(engine._pending) == 1

    monkeypatch.undo()
    assert engine.flush() is True
    with sqlite3.connect(str(tmp_path / "analytics.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 1


def test_no_storage_dir_keeps_the_previous_in_memory_behaviour(tmp_path):
    engine = AnalyticsEngine()
    _record(engine)
    assert engine.daily.requests == 1
    assert engine.flush() is True  # a no-op, not a failure


def test_an_unusable_store_degrades_instead_of_failing_requests(tmp_path, monkeypatch):
    """Analytics are reporting, not correctness. A broken analytics file must
    never take a request down with it."""
    monkeypatch.setattr(
        AnalyticsEngine, "_connect",
        lambda self: (_ for _ in ()).throw(sqlite3.DatabaseError("file is not a database")))

    engine = AnalyticsEngine(storage_dir=str(tmp_path))
    assert engine._db_path is None, "persistence should have been disabled"

    _record(engine)
    assert engine.daily.requests == 1


def test_corrupt_history_does_not_prevent_startup(tmp_path):
    (tmp_path / "analytics.db").write_bytes(b"this is not a sqlite file")
    engine = AnalyticsEngine(storage_dir=str(tmp_path))
    _record(engine)
    assert engine.daily.requests == 1


def test_the_proxy_wires_a_storage_dir(monkeypatch):
    """The engine supports persistence only if app.py actually passes the
    directory — the wiring is the part that silently regresses."""
    from tokenmizer.api import app as app_module
    assert app_module._analytics._db_path is not None, (
        "the proxy's analytics engine has no store; savings figures will "
        "reset on every restart"
    )
