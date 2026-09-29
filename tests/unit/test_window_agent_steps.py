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


# ── The cut has to hold for several requests, not one ────────────────────────

def test_a_cut_holds_across_several_requests_before_it_is_re_made(graph):
    """Cut to the ceiling and the next step goes over it, so the cut moves
    on every request and the provider's cache never serves the history.
    Cut to half of it and the tail has room to grow back."""
    window = SmartMessageWindow(token_budget=4000, protect_recent=8, max_tail_tokens=16000)
    conv = _agent_session(steps=14)
    held = window.apply(conv, graph, stable=True)[0]
    unchanged_for = 0
    for i in range(14, 24):
        conv = conv + _step(i)
        out = window.apply(conv, graph, stable=True)[0]
        _assert_valid(out)
        if out[:len(held)] == held:
            unchanged_for += 1
            held = out
        else:
            break
    assert unchanged_for >= 2, (
        f"the cut was re-made after {unchanged_for} request(s); it should "
        f"hold for several so the cached prefix is reused")


def test_a_recut_leaves_room_to_grow(graph):
    conv = _agent_session(steps=14)
    out, _ = SmartMessageWindow(
        token_budget=4000, protect_recent=8, max_tail_tokens=16000).apply(conv, graph)
    tail = [m for m in out if m.get("role") != "system"]
    assert count_messages_tokens(tail) <= 16000 // 2 + count_messages_tokens(_step(0)), (
        "a fresh cut should land at about half the ceiling, not just under it")


def test_chat_keeps_holding_a_cut_to_the_chat_budget(graph):
    """The tail ceiling is for agent loops. A chat session without tool
    results is held to the token budget as before, so a large ceiling does
    not quietly send four times as much history in a long chat."""
    conv = []
    for i in range(40):
        conv.append({"role": "user", "content": f"question {i} " + "words " * 50})
        conv.append({"role": "assistant", "content": f"answer {i} " + "words " * 50})
    conv.append({"role": "user", "content": "next"})
    small = SmartMessageWindow(token_budget=2000, protect_recent=8, max_tail_tokens=0)
    large = SmartMessageWindow(token_budget=2000, protect_recent=8, max_tail_tokens=60000)
    a = small.apply(conv, graph, stable=True)[0]
    b = large.apply(conv, graph, stable=True)[0]
    assert a == b
    for i in range(20):
        conv = conv + [{"role": "assistant", "content": "ok " + "words " * 50},
                       {"role": "user", "content": f"more {i} " + "words " * 50}]
        a = small.apply(conv, graph, stable=True)[0]
        b = large.apply(conv, graph, stable=True)[0]
        assert a == b, "a large tail ceiling changed how a plain chat is windowed"


# ── The index of paths seen in the output that is dropped ────────────────────
# Measured on a real agent session: 60% of what the agent went on to use and
# no longer had were file paths, and another third were parts of them. They
# were in a listing or a search result that windowing had dropped.

from tokenmizer.compression.window import _as_path, path_index  # noqa: E402


def _tool(text: str, call_id: str = "c1") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": text}


@pytest.mark.parametrize("word,expected", [
    ("tokenmizer/graph_memory/graph.py", "tokenmizer/graph_memory/graph.py"),
    ("./tests/unit/test_x.py", "tests/unit/test_x.py"),
    ("`api/app.py`,", "api/app.py"),
    ('"docs/roadmap.md"', "docs/roadmap.md"),
    ("pyproject.toml", "pyproject.toml"),
    ("/root/project/src/main.rs.", "/root/project/src/main.rs"),
    ("tokenmizer/checkpoints/", "tokenmizer/checkpoints/"),
])
def test_things_that_are_paths(word, expected):
    assert _as_path(word) == expected


@pytest.mark.parametrize("word", [
    "https://example.com/a/b.py",       # a URL
    "3.11", "1.5.0", "12/25",            # a version, a date
    "--max-tokens", "-v",                # flags
    "self.assertEqual", "cache.get",     # attribute access, not a file
    "the", "ok", "",                     # too short or plain words
    "a" * 300,                           # a blob
    "key=value/with=equals",             # not a path character set
])
def test_things_that_are_not_paths(word):
    assert _as_path(word) == ""


def test_the_index_is_grouped_by_directory_like_a_listing():
    out = path_index([_tool("edited tokenmizer/api/app.py and tokenmizer/api/routes.py "
                            "then tests/unit/test_a.py")], budget=500)
    assert "tokenmizer/api/: app.py routes.py" in out
    assert "tests/unit/: test_a.py" in out
    assert out.startswith("Paths seen in earlier tool output")


def test_paths_in_the_calls_count_too():
    call = {"role": "assistant", "content": None, "tool_calls": [{
        "id": "c1", "type": "function",
        "function": {"name": "read", "arguments": '{"file_path": "src/deep/module.py"}'}}]}
    assert "src/deep/: module.py" in path_index([call], budget=500)


def test_a_path_is_listed_once_however_often_it_appears():
    out = path_index([_tool("api/app.py " * 50), _tool("api/app.py")], budget=500)
    assert out.count("app.py") == 1


def test_nothing_to_index_is_an_empty_string():
    assert path_index([_tool("all tests passed, nothing else here")], budget=500) == ""
    assert path_index([], budget=500) == ""


def test_the_budget_holds_and_the_newest_directories_are_kept():
    old = _tool(" ".join(f"old{i}/file.py" for i in range(200)))
    new = _tool("newest/dir/keep.py")
    out = path_index([old, new], budget=120)
    from tokenmizer.core.tokenizer import count_tokens
    assert count_tokens(out) <= 120
    assert "newest/dir/: keep.py" in out


def test_a_budget_too_small_for_one_line_gives_no_index():
    assert path_index([_tool("api/app.py")], budget=3) == ""


@pytest.mark.parametrize("blob", [
    "a" * 400_000,                                   # one enormous word
    "a/" * 200_000,                                  # slashes all the way
    ("x" * 150 + "/") * 3000,                        # long segments
    ("1234567890" * 10 + " ") * 4000,                # many long words
    "/" * 400_000,
    ".-" * 200_000,
])
def test_extraction_is_linear_on_hostile_output(blob):
    """Tool output has minified lines and blobs. Every word is read once and
    long ones are skipped unread, so nothing here can be quadratic."""
    import time
    t0 = time.perf_counter()
    path_index([_tool(blob)], budget=1500)
    assert time.perf_counter() - t0 < 3.0


def test_extraction_reads_no_more_than_the_scan_cap():
    """A megabyte of output is not read past the cap, so a path that only
    appears after it is not found. Bounded work beats completeness here."""
    hidden = "x " * 250_000 + "late/only.py"
    assert path_index([_tool(hidden)], budget=500) == ""


# ── In the bridge ────────────────────────────────────────────────────────────

def _bridge(out):
    return next(m["content"] for m in out if m.get("role") == "system")


def _session_with_paths(steps=12):
    conv = [{"role": "user", "content": "Fix the failing login test in api/auth.py"}]
    for i in range(steps):
        conv += _step(i)
    conv[2]["content"] = ("src/only_in_the_first_result/needle.py " + "filler " * 3000)
    return conv


def test_paths_from_dropped_output_reach_the_bridge(graph):
    conv = _session_with_paths()
    out, _ = SmartMessageWindow(token_budget=4000, protect_recent=8,
                                max_tail_tokens=6000, tool_index_tokens=800).apply(conv, graph)
    assert "src/only_in_the_first_result/: needle.py" in _bridge(out)
    assert all("needle.py" not in str(m.get("content")) for m in out
               if m.get("role") != "system"), "sanity: the output itself is gone"


def test_the_index_is_fenced_because_tool_output_is_not_trusted(graph):
    from tokenmizer.security.fencing import FENCE_OPEN

    out, _ = SmartMessageWindow(token_budget=4000, protect_recent=8, max_tail_tokens=6000,
                                tool_index_tokens=800).apply(_session_with_paths(), graph)
    bridge = _bridge(out)
    assert FENCE_OPEN in bridge
    assert bridge.index("paths seen in earlier tool output") < bridge.index("needle.py")


def test_a_budget_of_zero_leaves_the_index_out(graph):
    out, _ = SmartMessageWindow(token_budget=4000, protect_recent=8, max_tail_tokens=6000,
                                tool_index_tokens=0).apply(_session_with_paths(), graph)
    assert "needle.py" not in _bridge(out)


def test_chat_gets_no_index(graph):
    conv = []
    for i in range(30):
        conv.append({"role": "user", "content": f"look at docs/page{i}.md " + "words " * 60})
        conv.append({"role": "assistant", "content": "ok " + "words " * 60})
    conv.append({"role": "user", "content": "next"})
    out, _ = SmartMessageWindow(token_budget=2000, protect_recent=8,
                                tool_index_tokens=800).apply(conv, graph)
    assert "Paths seen" not in _bridge(out), "the index is for dropped tool output only"


def test_the_index_survives_a_held_cut(graph):
    """It is part of the frozen bridge: built at the cut and reused, so it
    costs once per cut and the provider's cache serves it after."""
    window = SmartMessageWindow(token_budget=4000, protect_recent=8,
                                max_tail_tokens=16000, tool_index_tokens=800)
    conv = _session_with_paths(steps=14)
    first, _ = window.apply(conv, graph, stable=True)
    second, _ = window.apply(conv + _step(14), graph, stable=True)
    assert "needle.py" in _bridge(first)
    assert _bridge(second) == _bridge(first)
    assert second[:len(first)] == first
