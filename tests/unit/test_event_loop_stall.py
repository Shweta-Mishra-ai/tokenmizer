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

These tests measure loop lateness, which is the thing that matters and is
also the thing that is robust: the ceiling is set well above the observed
value and far below the pre-fix one, so only a return to inline execution
trips it.
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
    @pytest.mark.parametrize("multiplier,ceiling_ms", [
        (24, 20.0),      # ~3 KB turn: was 33 ms
        (96, 25.0),      # ~12 KB pasted log: was 86 ms
    ])
    async def test_a_large_turn_does_not_stall_the_loop(
            self, multiplier, ceiling_ms):
        graph = GraphMemory("stall", storage_dir=tempfile.mkdtemp())
        messages = [{"role": "assistant", "content": _TURN * multiplier}]

        stall = await _worst_stall_while(
            app_module._extract_heuristic(graph, messages))

        assert graph._nodes, "the extraction must still have done its work"
        assert stall < ceiling_ms, (
            f"the loop was held for {stall:.1f} ms by a "
            f"{len(_TURN) * multiplier / 1000:.0f} KB turn — every "
            f"concurrent request pays that as latency"
        )

    async def test_the_inline_call_is_what_used_to_stall(self):
        """The same work, called inline, still blocks — so the test above
        is measuring the dispatch and not something incidental."""
        graph = GraphMemory("inline", storage_dir=tempfile.mkdtemp())
        messages = [{"role": "assistant", "content": _TURN * 96}]

        async def inline():
            graph.extract_from_messages(messages, incremental=True)

        assert await _worst_stall_while(inline()) > 20.0, (
            "the inline call no longer stalls the loop, so this file is "
            "measuring the wrong thing — re-derive the numbers"
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
