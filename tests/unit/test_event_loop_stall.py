"""
TokenMizer is a proxy. CPU it spends without yielding is latency every
OTHER in-flight request pays.

The heuristic extraction pass is the most expensive thing it does per
turn, and it is pure CPU — regex over the whole message, no awaits. Called
inline from the request handler it held the event-loop thread for as long
as it ran. Measured against a 0.23 ms idle floor:

    turn size      loop stall (before)   (after)
    ~120 B                    ~4 ms       <1 ms
    ~3 KB                     33 ms       ~6 ms
    ~12 KB pasted log         86 ms       ~6 ms

A worker thread does not make extraction faster — the GIL is held
throughout — but it lets the loop be scheduled between bytecodes rather
than waiting for the whole pass, so one caller pasting a log stops adding
its full extraction time to everyone else's latency.

What makes the thread safe is the session lock the request path already
holds around `_update_graph`, NOT the older argument that GraphMemory's
mutators contain no `await`. That argument is about coroutines and says
nothing about threads; the comment making it has been corrected.

These tests measure loop lateness, which is the thing that matters. They
compare the two dispatches on the same machine rather than against a
fixed millisecond ceiling: the figures above are Linux, and the Windows
runner stalls ~37 ms threaded where it would stall far more inline. A
ceiling calibrated on one platform fails on another for reasons that have
nothing to do with the behaviour under test.
"""
from __future__ import annotations

import asyncio
import inspect
import tempfile
import time

import pytest

from tokenmizer.api import app as app_module
from tokenmizer.graph_memory.graph import GraphMemory

_TURN = ("Add rate limiting to the API. Decided: use a token bucket in "
         "Redis. Files: src/api/limits.py, src/api/deps.py. "
         "Error: TimeoutError when the pool is exhausted. ")


async def _worst_stall_while(coro) -> float:
    """Milliseconds the event loop was held while `coro` ran.

    A ticker wakes every 2 ms and records how late it was; lateness is
    exactly time some other task held the thread without awaiting.
    """
    late: list[float] = []
    running = True

    async def ticker():
        while running:
            started = time.perf_counter()
            await asyncio.sleep(0.002)
            late.append((time.perf_counter() - started - 0.002) * 1000)

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0.05)                    # settle, and prove the floor
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


class TestExtractionDoesNotHoldTheEventLoop:
    """Compares the two dispatches ON THE SAME MACHINE, in the same run.

    An absolute ceiling was the first attempt and it was wrong: calibrated
    from Linux (3 ms threaded, 86 ms inline), it failed on the Windows
    runner at 37 ms threaded — where the inline figure is proportionally
    higher too, so the fix was working and only the yardstick was
    provincial. Scheduler granularity, `to_thread` overhead and runner
    contention all differ per platform, and none of them is what this test
    is about.

    The claim is relative and so is the measurement: dispatching to a
    thread must stall the loop materially less than running inline. That
    holds on any machine without a number baked in from one of them.
    """

    @pytest.mark.parametrize("multiplier", [24, 96])   # ~3 KB and ~12 KB turns
    async def test_the_thread_stalls_the_loop_far_less_than_inline(
            self, multiplier):
        messages = [{"role": "assistant", "content": _TURN * multiplier}]

        inline_graph = GraphMemory("inline", storage_dir=tempfile.mkdtemp())

        async def inline():
            inline_graph.extract_from_messages(messages, incremental=True)

        inline_stall = await _worst_stall_while(inline())

        threaded_graph = GraphMemory("threaded", storage_dir=tempfile.mkdtemp())
        threaded_stall = await _worst_stall_while(
            app_module._extract_heuristic(threaded_graph, messages))

        assert threaded_graph._nodes, "the extraction must still do its work"
        assert inline_stall > 5.0, (
            f"inline extraction only stalled {inline_stall:.1f} ms, so this "
            f"payload is too small to measure a difference — raise the "
            f"multiplier rather than trusting the comparison below"
        )
        assert threaded_stall < inline_stall / 3, (
            f"dispatching to a thread stalled the loop {threaded_stall:.1f} ms "
            f"against {inline_stall:.1f} ms inline — not the order-of-"
            f"magnitude difference the thread exists for"
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
