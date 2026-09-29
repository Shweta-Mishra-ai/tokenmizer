"""
Windowing in an agent loop.

`protect_recent` counts messages, and in a tool loop one message is a file
or a test log. On a real agent session the window opened on a user turn
(there were two in fifty messages), so the "protected" tail was nearly the
whole conversation and windowing removed nothing at all. The tail is now
measured in tokens, cut only at the start of a step, and the newest step is
never dropped.

The hard requirement is validity. Both providers reject a tool result that
is not preceded by the assistant turn that called it, and a conversation
that does not open on a user turn, so every test that drops turns also
checks those two invariants.
"""
from __future__ import annotations

import pytest

from tokenmizer.compression.window import SmartMessageWindow
from tokenmizer.core.tokenizer import count_messages_tokens
from tokenmizer.graph_memory.graph import GraphMemory


@pytest.fixture
def graph(tmp_path):
    g = GraphMemory("agent-window", storage_dir=str(tmp_path))
    g.extract_from_messages([
        {"role": "user", "content": "Fix the failing login test in api/auth.py"},
        {"role": "assistant", "content": "Decided: reproduce first. Working on api/auth.py."},
    ], incremental=False)
    return g


def _step(i: int, size: int = 400, calls: int = 1) -> list[dict]:
    """One agent step: an assistant turn calling `calls` tools, and results."""
    ids = [f"call_{i}_{k}" for k in range(calls)]
    out = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": cid, "type": "function",
         "function": {"name": "read", "arguments": f'{{"path": "f{i}_{k}.py"}}'}}
        for k, cid in enumerate(ids)]}]
    for k, cid in enumerate(ids):
        out.append({"role": "tool", "tool_call_id": cid,
                    "content": f"output {i}.{k} " + ("line of file content " * size)})
    return out


def _agent_session(steps: int, size: int = 400, calls: int = 1) -> list[dict]:
    msgs = [{"role": "user", "content": "Fix the failing login test in api/auth.py"}]
    for i in range(steps):
        msgs += _step(i, size, calls)
    return msgs


def _assert_valid(conv: list[dict]) -> None:
    body = [m for m in conv if m.get("role") != "system"]
    assert body[0]["role"] == "user", "a conversation must open on a user turn"
    answered: set[str] = set()
    for m in body:
        if m["role"] == "assistant":
            answered = {c["id"] for c in m.get("tool_calls") or []}
        elif m["role"] == "tool":
            assert m["tool_call_id"] in answered, (
                f"tool result {m['tool_call_id']} is not preceded by the "
                f"assistant turn that called it")
        else:
            answered = set()


def _call_ids(conv):
    return [c["id"] for m in conv for c in (m.get("tool_calls") or [])]


# ── The failure this exists for ──────────────────────────────────────────────

def test_a_token_ceiling_shrinks_a_tail_that_message_counting_cannot(graph):
    """Ten messages of file contents are most of the payload, so counting
    messages leaves the tail huge. The ceiling is what shrinks it."""
    conv = _agent_session(steps=12)
    by_count, saved_count = SmartMessageWindow(
        token_budget=4000, protect_recent=10).apply(conv, graph)
    by_tokens, saved_tokens = SmartMessageWindow(
        token_budget=4000, protect_recent=10, max_tail_tokens=6000).apply(conv, graph)
    assert saved_tokens > saved_count
    assert count_messages_tokens(by_tokens) < count_messages_tokens(by_count)
    assert count_messages_tokens(by_count) > 0.4 * count_messages_tokens(conv), (
        "sanity: message counting alone leaves most of the payload")


def test_the_ceiling_drops_whole_steps_and_keeps_the_conversation_valid(graph):
    conv = _agent_session(steps=12)
    out, saved = SmartMessageWindow(
        token_budget=4000, protect_recent=10, max_tail_tokens=6000).apply(conv, graph)
    assert saved > 0
    _assert_valid(out)


def test_the_newest_step_is_never_dropped_however_large(graph):
    conv = _agent_session(steps=6) + _step(6, size=3000)     # one huge, newest
    out, _ = SmartMessageWindow(
        token_budget=1000, protect_recent=4, max_tail_tokens=500).apply(conv, graph)
    _assert_valid(out)
    assert out[-1] == conv[-1], "the result the model is about to read was cut"
    assert "call_6_0" in _call_ids(out)


def test_a_parallel_batch_stays_together(graph):
    """Several calls in one assistant turn and all their results are one
    step; splitting them strands results from their call."""
    conv = _agent_session(steps=6, calls=3)
    out, _ = SmartMessageWindow(
        token_budget=1000, protect_recent=4, max_tail_tokens=2500).apply(conv, graph)
    _assert_valid(out)
    for m in out:
        if m.get("tool_calls"):
            ids = {c["id"] for c in m["tool_calls"]}
            answered = {x["tool_call_id"] for x in out if x.get("role") == "tool"}
            assert ids <= answered, "a call in the tail lost a result"


def test_a_lead_in_opens_the_conversation_when_the_window_starts_on_a_step(graph):
    conv = _agent_session(steps=12)
    out, _ = SmartMessageWindow(
        token_budget=4000, protect_recent=10, max_tail_tokens=6000).apply(conv, graph)
    body = [m for m in out if m.get("role") != "system"]
    assert body[0]["role"] == "user" and "summarised" in body[0]["content"]
    assert body[1]["role"] == "assistant"


def test_no_lead_in_when_the_window_opens_on_a_real_user_turn(graph):
    conv = _agent_session(steps=3) + [{"role": "user", "content": "now also fix logout"}]
    conv += _step(3) + _step(4)
    out, _ = SmartMessageWindow(
        token_budget=1000, protect_recent=6).apply(conv, graph)
    body = [m for m in out if m.get("role") != "system"]
    assert body[0] == {"role": "user", "content": "now also fix logout"}


# ── Chat is unchanged ────────────────────────────────────────────────────────

def test_a_plain_chat_is_cut_exactly_as_before(graph):
    conv = []
    for i in range(30):
        conv.append({"role": "user", "content": f"question {i} " + "words " * 60})
        conv.append({"role": "assistant", "content": f"answer {i} " + "words " * 60})
    conv.append({"role": "user", "content": "and the last one"})
    out, _ = SmartMessageWindow(token_budget=2000, protect_recent=8).apply(conv, graph)
    body = [m for m in out if m.get("role") != "system"]
    assert body[0]["role"] == "user"
    assert 1 <= len(body) <= 9
    assert body[-1] == conv[-1]
    assert not any("summarised" in str(m.get("content")) for m in body)


def test_a_ceiling_of_zero_means_no_ceiling(graph):
    conv = _agent_session(steps=12)
    a, _ = SmartMessageWindow(token_budget=4000, protect_recent=10).apply(conv, graph)
    b, _ = SmartMessageWindow(token_budget=4000, protect_recent=10,
                              max_tail_tokens=0).apply(conv, graph)
    assert a == b


# ── The frozen cut still works with a ceiling ────────────────────────────────

def test_the_cut_holds_between_cuts_in_an_agent_loop(graph):
    """Otherwise every request re-cuts and the provider's cache never
    serves the history — the reason the cut is frozen at all."""
    window = SmartMessageWindow(token_budget=4000, protect_recent=8, max_tail_tokens=16000)
    conv = _agent_session(steps=14)
    first, _ = window.apply(conv, graph, stable=True)
    conv2 = conv + _step(14)
    second, _ = window.apply(conv2, graph, stable=True)
    _assert_valid(second)
    assert second[:len(first)] == first, "the second request must extend the first"


def test_the_frozen_cut_is_dropped_when_the_tail_outgrows_the_ceiling(graph):
    window = SmartMessageWindow(token_budget=4000, protect_recent=8, max_tail_tokens=6000)
    conv = _agent_session(steps=12)
    first, _ = window.apply(conv, graph, stable=True)
    grown = conv + [m for i in range(12, 22) for m in _step(i)]
    second, _ = window.apply(grown, graph, stable=True)
    _assert_valid(second)
    assert count_messages_tokens(second) < count_messages_tokens(first) * 3


# ── What is promoted into the bridge ─────────────────────────────────────────

def test_tool_output_is_never_promoted_to_a_session_constraint(graph):
    """Tool results are windowed out with everything else, and a log line
    that says "must be under 5s" is not something the user asked for."""
    from tokenmizer.graph_memory.summary import select_sentences

    dropped = [
        {"role": "tool", "tool_call_id": "c",
         "content": "Timeout must be under 5 seconds. The limit is 30 requests."},
        {"role": "user", "content": "The bundle must stay under 500KB, no exceptions."},
    ]
    kept = select_sentences(dropped, known=[])
    assert any("500KB" in k for k in kept)
    assert not any("30 requests" in k or "5 seconds" in k for k in kept)


# ── What the provider actually receives ──────────────────────────────────────

def test_the_anthropic_request_built_from_a_windowed_agent_loop_is_valid(graph):
    """Anthropic's rules, checked on the converted request rather than the
    OpenAI-shaped one: it opens on a user turn, every tool_use is answered
    by a tool_result in the very next message, and no tool_result stands
    without its tool_use."""
    from tokenmizer.providers.providers import conversation_messages
    from tokenmizer.providers.tools import anthropic_messages

    conv = _agent_session(steps=14, calls=2)
    out, _ = SmartMessageWindow(
        token_budget=4000, protect_recent=8, max_tail_tokens=6000).apply(conv, graph)
    wire = anthropic_messages(conversation_messages(out))

    assert wire[0]["role"] == "user"
    for i, m in enumerate(wire):
        blocks = m["content"] if isinstance(m["content"], list) else []
        uses = [b["id"] for b in blocks if b.get("type") == "tool_use"]
        results = [b["tool_use_id"] for b in blocks if b.get("type") == "tool_result"]
        if uses:
            nxt = wire[i + 1]["content"]
            answered = [b["tool_use_id"] for b in nxt if b.get("type") == "tool_result"]
            assert sorted(answered) == sorted(uses), f"tool_use unanswered at message {i}"
        if results:
            prev = wire[i - 1]["content"]
            called = [b["id"] for b in prev if b.get("type") == "tool_use"]
            assert sorted(results) == sorted(called), f"orphan tool_result at message {i}"
