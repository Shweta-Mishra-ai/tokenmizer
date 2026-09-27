"""
TokenMizer is a proxy. CPU it spends without yielding is latency every
OTHER in-flight request pays.

The heuristic extraction pass is the most expensive thing it does per
turn, and it is pure CPU — regex over the whole message, no awaits.
Called inline from the request handler it held the event-loop thread for
as long as it ran. It now runs in a worker thread
(`app._extract_heuristic`).

HOW MUCH THAT HELPS IS MACHINE-DEPENDENT, and the honest numbers are
worth writing down because the first version of this file asserted an
improvement that does not hold everywhere:

    machine                       inline      threaded
    4-core container              154 ms       6.5 ms   (25x better)
    windows-latest runner          74 ms      30 ms     (2.5x better)
    ubuntu-latest 3.10 runner     114 ms     114 ms     (no better)

The reason is that `re` does not release the GIL, and CPython hands it
over only BETWEEN bytecodes — a single `re.search` over a large string is
one bytecode and holds the lock for its whole duration. So the loop can
only be scheduled between regex calls, and the benefit is bounded by how
long the longest single call takes on that hardware. Verified directly:
60 back-to-back searches took 439 ms and starved another thread for
13.4 ms, about one call's worth.

The thread is therefore an improvement where extraction is many short
calls and a no-op where one call dominates. It is never WORSE, which is
the one timing property that holds on every machine and the only one
asserted here. The rest of this file is structural and deterministic.
"""
from __future__ import annotations

import asyncio
import inspect
import time

import pytest

from tokenmizer.api import app as app_module
from tokenmizer.graph_memory.graph import GraphMemory

_TURN = ("Add rate limiting to the API. Decided: use a token bucket in "
         "Redis. Files: src/api/limits.py, src/api/deps.py. "
         "Error: TimeoutError when the pool is exhausted. ")


async def _idle_floor() -> float:
    """Worst tick lateness with nothing else running.

    This is not zero and on some platforms it is not small. `sleep(0.002)`
    is quantised to the system timer, which on Windows is about 15.6 ms,
    so every tick is ~13 ms "late" on a completely idle loop. That floor
    lands on BOTH readings and crushes the ratio between them: a true 4x
    difference measured 30.3 ms against 74.1 ms, which is 2.4x, and the
    test failed for a reason that had nothing to do with the code.
    """
    return await _worst_stall_while(asyncio.sleep(0.03))


async def _worst_stall_while(coro) -> float:
    """Milliseconds the event loop was held while `coro` ran.

    A ticker wakes every 2 ms and records how late it was; lateness is
    exactly time some other task held the thread without awaiting. Callers
    subtract `_idle_floor()` to get the part attributable to `coro`.
    """
    late: list[float] = []
    running = True

    async def ticker():
        while running:
            started = time.perf_counter()
            await asyncio.sleep(0.002)
            late.append((time.perf_counter() - started - 0.002) * 1000)

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0.05)                    # settle
    late.clear()
    try:
        await coro
    finally:
        # The ticker cannot record a stall WHILE it is being starved — it
        # records on its next wake. Give it that wake before stopping,
        # or a fully blocking call reports zero lateness, which is the
        # opposite of the truth.
        await asyncio.sleep(0.005)
        running = False
        task.cancel()
    assert late, "the ticker never ran; the measurement is meaningless"
    return max(late)


class TestTheThreadIsNeverWorseThanInline:
    """The only timing claim that survives every platform.

    "Materially better" does not: see the table in the module docstring —
    one runner measured 114 ms both ways. Asserting an improvement there
    fails for a reason that is true about the GIL rather than about this
    code, which is how the first version of this test turned `main` red.

    What a thread must not do is make things worse. The dispatch costs a
    thread hop, and if that ever grew into something comparable to the
    work itself this would catch it.
    """

    @pytest.mark.parametrize("multiplier", [96, 240])   # ~12 KB and ~30 KB
    async def test_dispatching_to_a_thread_does_not_increase_the_stall(
            self, multiplier, tmp_path):
        messages = [{"role": "assistant", "content": _TURN * multiplier}]
        floor = await _idle_floor()

        inline_graph = GraphMemory("inline", storage_dir=str(tmp_path / "i"))

        async def inline():
            inline_graph.extract_from_messages(messages, incremental=True)

        inline_stall = max(await _worst_stall_while(inline()) - floor, 0.01)

        threaded_graph = GraphMemory("threaded", storage_dir=str(tmp_path / "t"))
        threaded_stall = max(
            await _worst_stall_while(
                app_module._extract_heuristic(threaded_graph, messages)) - floor,
            0.0)

        assert threaded_graph._nodes, "the extraction must still do its work"
        # 1.5x, not 1.0x: both readings carry scheduler noise, and the
        # claim is "not materially worse", not "never a microsecond more".
        assert threaded_stall < inline_stall * 1.5, (
            f"the thread stalled the loop {threaded_stall:.1f} ms against "
            f"{inline_stall:.1f} ms inline (floor {floor:.1f} ms) — the "
            f"dispatch has become a cost rather than a saving"
        )


class TestTheDispatchLivesInOnePlace:
    def test_no_request_path_extraction_bypasses_the_helper(self):
        """There were four identical inline copies of this call. A helper
        that three of them ignore is worse than no helper."""
        source = inspect.getsource(app_module)
        stripped = "\n".join(
            ln for ln in source.splitlines() if not ln.lstrip().startswith("#")
        )
        assert "graph.extract_from_messages(raw_messages" not in stripped, (
            "a request-path extraction call bypasses _extract_heuristic "
            "and will stall the event loop"
        )

    def test_the_helper_dispatches_to_a_thread(self):
        source = inspect.getsource(app_module._extract_heuristic)
        assert "asyncio.to_thread" in source
