"""
The rate limit every worker shares.

`RateLimiter` keeps its token buckets in a process dict, which is correct
for one process and wrong for the deployment the Dockerfile ships: with
`--workers 4` a configured 60 requests per minute is enforced four times
over, once per worker, and the operator gets 240. A limit that is not the
limit is worse than no limit, because it is written down.

`state_backend: sqlite` puts the buckets where every worker on the host
can see them. The test that matters is the last one in this file: it
spends the budget from several REAL processes, which is the only way to
show the lost update is gone — two workers reading the last token,
both deciding they may spend it, and both allowing the request.

What this is not, and the module says so too: a distributed limiter.
SQLite is shared through the filesystem, so it is shared by the workers
on one host. Several hosts behind a load balancer need the limit at the
load balancer.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from tokenmizer.api.rate_limiter import RateLimiter
from tokenmizer.api.shared_rate_limiter import SQLiteRateLimiter


@pytest.fixture
def limiter(tmp_path):
    return SQLiteRateLimiter(rate=6, per_seconds=60, burst=0,
                             storage_dir=str(tmp_path))


class TestItBehavesLikeTheOneItReplaces:

    async def test_allows_up_to_capacity_then_refuses(self, limiter):
        for i in range(6):
            allowed, retry = await limiter.check("client")
            assert allowed, f"refused request {i + 1} of a budget of 6"
            assert retry == 0.0

        allowed, retry = await limiter.check("client")
        assert not allowed
        assert retry > 0, "a refusal must say when to come back"

    async def test_clients_have_their_own_budgets(self, limiter):
        for _ in range(6):
            assert (await limiter.check("noisy"))[0]
        assert not (await limiter.check("noisy"))[0]

        assert (await limiter.check("quiet"))[0], (
            "one client exhausting its budget must not refuse another"
        )

    async def test_the_interface_matches_the_in_process_limiter(self, limiter):
        """app.py holds one or the other and must not know which."""
        other = RateLimiter(rate=6, per_seconds=60, burst=0)

        assert type(await limiter.check("x")) is type(await other.check("x"))


class TestItFailsOpenAndSaysSo:
    """A limiter that fails closed turns a locked database file into a
    total outage. This one lets traffic through and logs the fault — the
    same trade the graph makes when persistence breaks."""

    async def test_an_unopenable_store_falls_back_to_allowing(self, tmp_path, caplog):
        import logging

        blocked = tmp_path / "not-a-directory"
        blocked.write_text("")          # a FILE where the dir should be

        with caplog.at_level(logging.ERROR):
            limiter = SQLiteRateLimiter(storage_dir=str(blocked / "sub"))

        assert limiter.available is False
        assert (await limiter.check("anyone"))[0] is True
        assert any("rate limiter" in r.message.lower() for r in caplog.records)

    async def test_a_broken_check_allows_the_request(self, limiter, monkeypatch, caplog):
        import logging

        def boom(*a, **k):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(limiter, "_connect", boom)

        with caplog.at_level(logging.ERROR):
            allowed, retry = await limiter.check("client")

        assert allowed is True and retry == 0.0
        assert any("rate limiter" in r.message.lower() for r in caplog.records), (
            "failing open silently is the one thing worse than failing open"
        )


class TestTheProxyPicksTheRightOne:

    def test_memory_is_the_default(self):
        from tokenmizer.config.settings import Settings

        assert Settings().state_backend == "memory"

    def test_sqlite_selects_the_shared_limiter(self, tmp_path, monkeypatch):
        from tokenmizer.api import app as app_module

        monkeypatch.setattr(app_module.settings, "state_backend", "sqlite")
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "storage_dir",
                            str(tmp_path))

        assert isinstance(app_module._build_rate_limiter(), SQLiteRateLimiter)

    def test_memory_selects_the_in_process_limiter(self, monkeypatch):
        from tokenmizer.api import app as app_module

        monkeypatch.setattr(app_module.settings, "state_backend", "memory")

        assert isinstance(app_module._build_rate_limiter(), RateLimiter)

    def test_redis_is_accepted_but_warns_that_it_does_nothing(
            self, tmp_path, monkeypatch, caplog):
        import logging

        import tokenmizer.config.settings as settings_module

        cfg = tmp_path / "tokenmizer.yaml"
        cfg.write_text("state_backend: redis\n")
        monkeypatch.setenv("TOKENMIZER_CONFIG", str(cfg))
        monkeypatch.setattr(settings_module, "_settings", None)
        monkeypatch.delenv("TOKENMIZER_ENV", raising=False)
        # conftest pins this for the whole suite, and an env var
        # deliberately outranks the YAML — so the YAML under test would
        # otherwise be dropped before it is ever read.
        monkeypatch.delenv("TOKENMIZER_STATE_BACKEND", raising=False)

        with caplog.at_level(logging.WARNING, logger="tokenmizer.config.settings"):
            loaded = settings_module.get_settings()
        monkeypatch.setattr(settings_module, "_settings", None)

        assert loaded.state_backend == "redis", "the value must still load"
        assert any("NOT IMPLEMENTED" in r.message and "sqlite" in r.message
                   for r in caplog.records), [r.message for r in caplog.records]


# The one that justifies the module. Separate processes, not tasks: an
# asyncio lock would hide exactly the bug this exists to fix.
_WORKER = textwrap.dedent("""
    import asyncio, sys
    from tokenmizer.api.shared_rate_limiter import SQLiteRateLimiter

    async def main():
        limiter = SQLiteRateLimiter(rate=10, per_seconds=600, burst=0,
                                    storage_dir=sys.argv[1])
        allowed = 0
        for _ in range(10):
            ok, _retry = await limiter.check("shared-client")
            allowed += int(ok)
        print(allowed)

    asyncio.run(main())
""")


def test_four_processes_share_one_budget(tmp_path):
    """Four workers, ten requests each, one budget of ten.

    Per-process buckets would allow all forty. The assertion is exactly
    ten: not "fewer than forty", because a limiter that is merely
    stricter than broken is still not the configured limit.
    """
    script = tmp_path / "worker.py"
    script.write_text(_WORKER)

    procs = [
        subprocess.Popen([sys.executable, str(script), str(tmp_path)],
                         stdout=subprocess.PIPE, text=True)
        for _ in range(4)
    ]
    allowed = sum(int(p.communicate()[0].strip().splitlines()[-1]) for p in procs)

    assert allowed == 10, (
        f"four workers sharing a budget of 10 allowed {allowed} requests; "
        f"40 means the buckets are still per-process, and anything else "
        f"means two workers spent the same token"
    )
