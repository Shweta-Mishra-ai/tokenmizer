"""
What happens when every worker starts at the same moment.

`--workers 4` means four processes run `CREATE TABLE IF NOT EXISTS`
against the same file within milliseconds of each other, and
`PRAGMA journal_mode=WAL` takes a brief exclusive lock to switch mode. On
a loaded machine somebody loses that race and gets "database is locked".

Both SQLite-backed stores used to treat that first failure as final:
`available` was set once, in `__init__`, from one attempt. A worker that
lost the startup race therefore spent its **entire life** with the store
disabled — and for the rate limiter that means it silently enforced no
limit at all, which is the failure this whole module exists to prevent.

It was not caught by the four-process test in isolation. It showed up
only when the full suite ran alongside it and the machine was busy: 20
requests allowed against a budget of 10. That is the shape of a load bug
— invisible until something else is competing for the disk.

Three things are pinned here: losing the pragma race is not a failure, a
locked database at startup is retried rather than believed, and a store
that starts unavailable can recover instead of staying dead.
"""
from __future__ import annotations

import sqlite3
import subprocess
import sys
import textwrap
import time

import pytest

from tokenmizer.api.shared_rate_limiter import SQLiteRateLimiter
from tokenmizer.preferences import PreferenceStore

# Eight processes, not four: the bug needs contention to appear, and the
# machine running this is already busy with the rest of the suite.
_COLD_START = textwrap.dedent("""
    import asyncio, sys
    from tokenmizer.api.shared_rate_limiter import SQLiteRateLimiter

    async def main():
        limiter = SQLiteRateLimiter(rate=5, per_seconds=600, burst=0,
                                    storage_dir=sys.argv[1])
        # available is the whole point: a worker that lost the startup
        # race used to report False here and never rate-limit again.
        print("available" if limiter.available else "UNAVAILABLE")

    asyncio.run(main())
""")


def test_eight_workers_starting_at_once_all_get_a_working_store(tmp_path):
    script = tmp_path / "cold.py"
    script.write_text(_COLD_START)

    procs = [
        subprocess.Popen([sys.executable, str(script), str(tmp_path)],
                         stdout=subprocess.PIPE, text=True)
        for _ in range(8)
    ]
    results = [p.communicate()[0].strip().splitlines()[-1] for p in procs]

    assert results.count("UNAVAILABLE") == 0, (
        f"{results.count('UNAVAILABLE')} of 8 workers disabled their own "
        f"rate limiting because another worker held the file at startup"
    )


class _RefusingConnection:
    """A sqlite3 connection that raises on the statements matching
    `refuse`, and behaves normally otherwise.

    `sqlite3.Connection` is an immutable type, so its `execute` cannot be
    monkeypatched; wrapping the factory is the way to simulate a locked
    database without actually holding a lock for five seconds.
    """

    def __init__(self, inner, refuse, counter=None):
        self._inner = inner
        self._refuse = refuse
        self._counter = counter

    def execute(self, sql, *a, **k):
        if self._refuse(sql):
            if self._counter is not None:
                self._counter["n"] += 1
            raise sqlite3.OperationalError("database is locked")
        return self._inner.execute(sql, *a, **k)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *a):
        return self._inner.__exit__(*a)


def _patch_connect(monkeypatch, refuse, counter=None):
    real_connect = sqlite3.connect

    def fake_connect(*a, **k):
        return _RefusingConnection(real_connect(*a, **k), refuse, counter)

    monkeypatch.setattr(sqlite3, "connect", fake_connect)


class TestLosingThePragmaRaceIsNotAFailure:
    """`journal_mode=WAL` is a property of the FILE. Once any process has
    set it, it stays set — so failing to be the one that sets it says
    nothing about whether the store works."""

    @pytest.mark.parametrize("factory", [
        lambda d: SQLiteRateLimiter(storage_dir=str(d)),
        lambda d: PreferenceStore(storage_dir=str(d)),
    ])
    def test_a_locked_pragma_does_not_disable_the_store(self, factory,
                                                        tmp_path, monkeypatch):
        _patch_connect(monkeypatch, lambda sql: "journal_mode" in sql)

        assert factory(tmp_path).available is True


class TestALockedStartIsRetried:

    def test_init_retries_before_giving_up(self, tmp_path, monkeypatch):
        """Two locked attempts then success must end available, not dead."""
        seen = {"n": 0}
        _patch_connect(
            monkeypatch,
            lambda sql: (sql.strip().upper().startswith("CREATE TABLE")
                         and seen["n"] < 2),
            counter=seen,
        )
        limiter = SQLiteRateLimiter(storage_dir=str(tmp_path))

        assert limiter.available is True
        assert seen["n"] == 2, "it should have retried, not given up"

    def test_a_genuinely_broken_path_still_gives_up(self, tmp_path):
        """Retrying must not turn an unusable path into a hang."""
        blocked = tmp_path / "a-file"
        blocked.write_text("")

        started = time.monotonic()
        limiter = SQLiteRateLimiter(storage_dir=str(blocked / "sub"))
        elapsed = time.monotonic() - started

        assert limiter.available is False
        assert elapsed < 2.0, f"gave up only after {elapsed:.1f}s"


class TestAnUnavailableStoreCanRecover:

    async def test_it_retries_init_rather_than_staying_dead(self, tmp_path,
                                                            monkeypatch):
        blocked = tmp_path / "a-file"
        blocked.write_text("")
        limiter = SQLiteRateLimiter(rate=2, per_seconds=600, burst=0,
                                    storage_dir=str(blocked / "sub"))
        assert limiter.available is False

        # The path becomes usable — a sibling worker finished creating the
        # directory, the volume mounted, the disk came back.
        blocked.unlink()
        (blocked / "sub").mkdir(parents=True)
        # ...and enough time passes that the failure is no longer believed.
        limiter._unavailable_since -= 100

        assert (await limiter.check("client"))[0] is True
        assert limiter.available is True, (
            "a store that was locked at startup must be able to come back; "
            "the alternative is a worker that never rate-limits again"
        )

    async def test_a_still_broken_store_is_not_retried_every_request(
            self, tmp_path, monkeypatch):
        """Recovery must not become an open() on every single request."""
        blocked = tmp_path / "a-file"
        blocked.write_text("")
        limiter = SQLiteRateLimiter(storage_dir=str(blocked / "sub"))

        calls = {"n": 0}
        real_init = limiter._init

        def counting():
            calls["n"] += 1
            return real_init()

        monkeypatch.setattr(limiter, "_init", counting)

        for _ in range(20):
            await limiter.check("client")

        assert calls["n"] == 0, (
            "a failure inside the retry window must be believed, or a "
            "broken disk costs an open() per request"
        )


@pytest.mark.parametrize("store_factory", [
    lambda d: SQLiteRateLimiter(storage_dir=str(d)),
    lambda d: PreferenceStore(storage_dir=str(d)),
])
def test_both_stores_share_the_same_hardening(store_factory, tmp_path):
    """The two were written days apart with the same bug in both. If one
    grows a fix the other must have it too."""
    store = store_factory(tmp_path)

    assert store.available is True
    assert hasattr(store, "_init")
