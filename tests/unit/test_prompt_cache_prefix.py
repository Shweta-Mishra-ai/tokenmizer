"""
The proxy must not break the provider's prompt cache, and must use it.

Providers bill a prefix they have already seen at a fraction of the input
price, keyed on the exact bytes up to a cache marker, system prompt first.
Three things used to defeat that on every request:

1. The per-turn context block was appended to the system prompt, which is
   rendered before the conversation, so the first bytes of every request
   changed and nothing after them could be read from cache.
2. Once a session was windowed, the window slid forward one turn per
   request and rebuilt the bridge each time, so no two requests shared a
   prefix at all.
3. Only the system prompt was ever marked cacheable, never the history,
   and the per-model minimums were wrong in both directions.

These tests pin the request shape, not a price. benchmarks/savings prices
the same property end to end.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tokenmizer.compression.window import SmartMessageWindow
from tokenmizer.graph_memory.graph import GraphMemory
from tokenmizer.providers.providers import (
    _anthropic_input,
    _cache_minimum,
    _mark_history_cacheable,
)

MARK = {"type": "ephemeral"}


# ── Per-model minimums ───────────────────────────────────────────────────────

@pytest.mark.parametrize("model,minimum", [
    ("claude-opus-4-6", 4096),
    ("claude-opus-4-5", 4096),
    ("claude-haiku-4-5", 4096),
    ("claude-opus-4-7", 2048),
    ("claude-3-5-haiku-latest", 2048),
    ("claude-opus-4-8", 1024),
    ("claude-sonnet-5", 1024),
    ("claude-sonnet-4-6", 1024),
    ("claude-opus-4-1", 1024),
    ("claude-opus-5", 512),
    ("claude-fable-5-1", 512),
])
def test_cache_minimum_matches_the_documented_table(model, minimum):
    assert _cache_minimum(model) == minimum


def test_an_unknown_model_gets_the_smallest_minimum():
    """A marker under the real minimum is ignored at no cost; a threshold
    set too high silently forgoes the discount. So err low."""
    assert _cache_minimum("claude-something-new") == 512


# ── The history breakpoint ───────────────────────────────────────────────────

def test_marks_the_turn_before_the_one_being_asked():
    conv = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]
    out = _mark_history_cacheable(conv)
    assert out[1]["content"] == [{"type": "text", "text": "two", "cache_control": MARK}]
    assert out[2] == conv[2], "the last turn carries per-turn context; never marked"
    assert out[0] == conv[0]


def test_does_not_mutate_the_callers_messages():
    blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    conv = [{"role": "user", "content": blocks},
            {"role": "user", "content": "next"}]
    out = _mark_history_cacheable(conv)
    assert out[0]["content"][-1]["cache_control"] == MARK
    assert "cache_control" not in blocks[-1]
    assert conv[0]["content"] is blocks


@pytest.mark.parametrize("conv", [
    [],
    [{"role": "user", "content": "only"}],
    [{"role": "assistant", "content": "  "}, {"role": "user", "content": "q"}],
    [{"role": "assistant", "content": []}, {"role": "user", "content": "q"}],
])
def test_leaves_nothing_markable_alone(conv):
    assert _mark_history_cacheable(conv) == conv


def test_a_client_managing_its_own_cache_is_left_alone():
    """Its markers already say where its prefix is stable, and ours on top
    could exceed the provider's four-marker limit."""
    conv = [
        {"role": "user", "content": [{"type": "text", "text": "big doc",
                                      "cache_control": MARK}]},
        {"role": "assistant", "content": "read it"},
        {"role": "user", "content": "question"},
    ]
    assert _mark_history_cacheable(conv) == conv


def test_reported_prompt_includes_the_cached_part():
    """Anthropic's input_tokens is only the uncached tail. Reporting it
    alone would show a cached prompt as a small one, and the proxy would
    claim a saving the provider's cache made."""
    usage = SimpleNamespace(input_tokens=10, cache_read_input_tokens=900,
                            cache_creation_input_tokens=90)
    assert _anthropic_input(usage) == (1000, 900, 90)
    assert _anthropic_input(SimpleNamespace(input_tokens=7)) == (7, 0, 0)


# ── Stable windowing ─────────────────────────────────────────────────────────

def _session(turns: int) -> list[dict]:
    out = []
    for i in range(turns):
        out.append({"role": "user", "content": f"turn {i}: please add retries to "
                                               f"module_{i}.py and keep the tests green"})
        out.append({"role": "assistant", "content": f"Done: module_{i}.py now retries "
                                                    f"three times with backoff."})
    return out


@pytest.fixture
def graph(tmp_path):
    g = GraphMemory("stable-window", storage_dir=str(tmp_path))
    g.extract_from_messages(_session(10), incremental=False)
    return g


def _ask(window, graph, history, stable):
    return window.apply(history + [{"role": "user", "content": "next?"}],
                        graph, stable=stable)[0]


def test_consecutive_requests_share_everything_but_the_new_turns(graph):
    window = SmartMessageWindow(token_budget=400, protect_recent=4)
    history = _session(12)
    first = _ask(window, graph, history, stable=True)
    history += [{"role": "user", "content": "next?"},
                {"role": "assistant", "content": "ok"}]
    second = _ask(window, graph, history, stable=True)

    assert second[:len(first)] == first, (
        "the second request must start with the first one, byte for byte, "
        "or the provider re-bills the whole prefix"
    )
    assert second[len(first):] == history[-1:] + [{"role": "user", "content": "next?"}]


def test_the_sliding_window_is_what_broke_the_prefix(graph):
    """The same two requests without `stable`: the bridge and the first
    verbatim turn move, so the second request shares almost nothing."""
    window = SmartMessageWindow(token_budget=400, protect_recent=4)
    history = _session(12)
    first = _ask(window, graph, history, stable=False)
    history += [{"role": "user", "content": "next?"},
                {"role": "assistant", "content": "ok"}]
    second = _ask(window, graph, history, stable=False)
    assert second[:len(first)] != first


def test_recuts_when_the_verbatim_tail_outgrows_the_budget(graph):
    window = SmartMessageWindow(token_budget=400, protect_recent=4)
    history = _session(12)
    first = _ask(window, graph, history, stable=True)
    history += _session(30)[24:]            # a lot more conversation
    later = _ask(window, graph, history, stable=True)
    assert later[:len(first)] != first
    from tokenmizer.core.tokenizer import count_messages_tokens
    conv_after_bridge = [m for m in later if m.get("role") != "system"]
    assert len(conv_after_bridge) <= 4 + 1, "a fresh cut keeps protect_recent turns"
    assert count_messages_tokens(later) < count_messages_tokens(history)


def test_recuts_when_the_client_edits_history(graph, monkeypatch):
    """The frozen bridge describes the turns it replaced. If those turns
    are no longer the ones the request starts with, it no longer applies
    and must be rebuilt rather than reused."""
    window = SmartMessageWindow(token_budget=400, protect_recent=4)
    history = _session(12)
    builds = []
    original = graph.to_context_block
    monkeypatch.setattr(graph, "to_context_block",
                        lambda **kw: builds.append(1) or original(**kw))

    _ask(window, graph, history, stable=True)
    _ask(window, graph, history + [{"role": "user", "content": "next?"},
                                   {"role": "assistant", "content": "ok"}],
         stable=True)
    assert len(builds) == 1, "an unchanged history must reuse the cut"

    edited = [dict(m) for m in history]
    edited[0]["content"] = "a different opening question entirely"
    _ask(window, graph, edited, stable=True)
    assert len(builds) == 2, "an edited history reused a bridge built for other turns"


def test_nothing_is_dropped_that_the_sliding_window_keeps(graph):
    """Holding the cut still keeps MORE verbatim, never less."""
    history = _session(12) + [{"role": "user", "content": "next?"},
                              {"role": "assistant", "content": "ok"}]
    stable_w = SmartMessageWindow(token_budget=400, protect_recent=4)
    _ask(stable_w, graph, _session(12), stable=True)
    stable = _ask(stable_w, graph, history, stable=True)
    sliding = _ask(SmartMessageWindow(token_budget=400, protect_recent=4),
                   graph, history, stable=False)
    kept_sliding = [m for m in sliding if m.get("role") != "system"]
    kept_stable = [m for m in stable if m.get("role") != "system"]
    assert kept_stable[-len(kept_sliding):] == kept_sliding
