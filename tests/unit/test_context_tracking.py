"""
Regression tests for TM-04: context_pct must measure real payload occupancy,
not accumulate across turns.

Background — the bug this guards against:
  `_update_graph()` used to track occupancy with a stateful counter:
  `context_used = _get_context_used(session_id)` (read from the state
  backend), then `context_pct = (context_used + input_tokens) / window`,
  then `_set_context_used(session_id, context_used + input_tokens)`.

  Three problems with that design, independent of any concurrency bug:
    1. Each `messages` list already contains the FULL running conversation
       (that's the OpenAI-style contract) — so the tracked total double-
       counts every earlier turn's content on every subsequent turn,
       diverging further from real occupancy the longer a session runs.
    2. It never reflects windowing or compaction, so once the tracked
       value crossed `trigger_at_percent` it never came back down —
       every later turn re-triggered a full checkpoint write.
    3. The read-modify-write itself is a lost-update race under real
       concurrency (two requests for the same session interleaved across
       an `await` both read the same base value and one write clobbers
       the other) — reproduced separately in test_concurrency.py.

  The fix removes the accumulator entirely: `context_pct` is computed
  directly from the token count of what is actually about to be sent this
  turn. That is deterministic, race-free (no shared mutable state to race
  on), and — unlike the accumulator — actually reflects windowing.
"""
from __future__ import annotations

import pytest

from tokenmizer.api import app as app_module
from tokenmizer.graph_memory.graph import GraphMemory


@pytest.fixture
def graph(tmp_path):
    return GraphMemory("ctx-test-session", storage_dir=str(tmp_path))


def _messages(n_user_turns: int, words_per_turn: int = 20) -> list[dict]:
    """A synthetic conversation with n_user_turns user/assistant pairs."""
    msgs = []
    for i in range(n_user_turns):
        msgs.append({"role": "user", "content": " ".join([f"word{i}"] * words_per_turn)})
        msgs.append({"role": "assistant", "content": " ".join([f"reply{i}"] * words_per_turn)})
    return msgs


class TestContextPctIsNotAccumulated:

    async def test_many_small_identical_turns_never_accumulate_into_a_checkpoint(
        self, graph, monkeypatch
    ):
        """Each individual turn here occupies ~5% of the context window.
        Under the OLD accumulator design (`context_used + input_tokens`,
        persisted and re-read every call), repeating it enough times sums
        past `trigger_at_percent` even though no single real conversation
        ever got close. That must not happen: this is the core TM-04 bug,
        and the exact scenario the old design gets wrong."""
        monkeypatch.setattr(app_module, "_context_window", lambda model: 1000)
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "trigger_at_percent", 0.85)
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "enabled", True)

        for i in range(20):  # 20 * ~5% == 100% under the old accumulator
            small = _messages(1, words_per_turn=8)  # ~50-60 tokens, ~5-6% of window=1000
            raw = [dict(m) for m in small]
            _, status = await app_module._update_graph(
                "ctx-test-session", graph, raw, small, "claude-sonnet-4-6",
                {}, "just a short message here",
            )
            assert status["attempted"] is False, (
                f"auto-checkpoint fired on call #{i+1} from repeated small "
                f"turns — the context_pct accumulator bug is back (each "
                f"turn alone is ~5% of the window; it should never trigger)"
            )

    async def test_no_stateful_context_counter_helpers_remain(self):
        """Structural guard: the accumulator helpers must not come back.
        This is deliberately an implementation-detail check — the design
        decision (no shared mutable occupancy counter) is the point of
        TM-04, not merely its current symptom."""
        assert not hasattr(app_module, "_get_context_used"), (
            "_get_context_used() was removed as part of the TM-04 fix — "
            "its reintroduction likely means the accumulator race is back"
        )
        assert not hasattr(app_module, "_set_context_used"), (
            "_set_context_used() was removed as part of the TM-04 fix — "
            "its reintroduction likely means the accumulator race is back"
        )


class TestAutoCheckpointTriggersOnRealSize:

    async def test_checkpoint_fires_from_single_turn_real_size(self, graph, monkeypatch):
        """A context window small enough that ONE turn already exceeds
        trigger_at_percent must checkpoint on that very first call — no
        history of prior calls required."""
        monkeypatch.setattr(app_module, "_context_window", lambda model: 60)
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "trigger_at_percent", 0.5)
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "enabled", True)

        big_messages = _messages(5, words_per_turn=15)  # comfortably > 30 tokens
        raw = [dict(m) for m in big_messages]

        _, status = await app_module._update_graph(
            "ctx-test-session", graph, raw, big_messages, "claude-sonnet-4-6",
            {}, "does this session need a checkpoint right now",
        )
        assert status["attempted"] is True
        assert status["succeeded"] is True

    async def test_windowed_payload_can_drop_back_below_trigger(self, graph, monkeypatch):
        """The whole point of measuring the real payload instead of an
        accumulator: after windowing shrinks `messages`, context_pct must
        reflect the SMALLER post-windowing size, not a monotonically
        growing total. Use a window small enough that windowing kicks in
        (max_tokens_before_summary) and confirm the checkpoint decision is
        based on what's actually left after compaction, not on history."""
        monkeypatch.setattr(app_module, "_context_window", lambda model: 100_000)
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "trigger_at_percent", 0.85)
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "enabled", True)
        monkeypatch.setattr(app_module.settings.memory, "max_tokens_before_summary", 50)
        monkeypatch.setattr(app_module.settings.memory, "recent_turns_verbatim", 2)

        # Large conversation — windowing will fire and shrink it drastically
        # relative to the 100,000-token window, so real post-window pct is tiny.
        big = _messages(30, words_per_turn=20)
        raw = [dict(m) for m in big]
        _, status = await app_module._update_graph(
            "ctx-test-session", graph, raw, big, "claude-sonnet-4-6",
            {}, "summarize the whole conversation for me please",
        )
        assert status["attempted"] is False, (
            "checkpoint fired even though windowing should have reduced "
            "the payload to a small fraction of a 100,000-token window"
        )
