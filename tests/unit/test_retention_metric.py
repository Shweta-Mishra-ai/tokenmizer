"""
The retention figure in docs/benchmarks.md rests on a few small functions in
the benchmark runner, so their rules are pinned here: what counts as the
agent's "next step needs", when something counts as still present, and what
was actually sent. A metric nobody tested is a number nobody should quote.
"""
from __future__ import annotations

import pytest

from benchmarks.savings.runner import (
    _sent_text,
    load_trace_with_actions,
    needed_from_history,
    present,
    retention,
)

# ── What was sent ────────────────────────────────────────────────────────────


def test_sent_text_covers_a_plain_string_system_prompt_and_messages():
    text = _sent_text("be brief", [{"role": "user", "content": "hello there"}])
    assert "be brief" in text and "hello there" in text


def test_sent_text_covers_block_lists_and_a_missing_system_prompt():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "first block"},
                                     {"type": "text", "text": "second block"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1",
                                           "name": "read", "input": {"path": "a/b.py"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                      "content": "file body"}]},
    ]
    text = _sent_text(None, messages)
    for expected in ("first block", "second block", "a/b.py", "file body"):
        assert expected in text


def test_sent_text_covers_a_list_system_prompt():
    text = _sent_text([{"type": "text", "text": "cached prefix"}],
                      [{"role": "user", "content": "q"}])
    assert "cached prefix" in text


# ── Is a token still there ───────────────────────────────────────────────────

@pytest.mark.parametrize("token,text,expected", [
    ("api/app.py", "edited api/app.py today", True),
    ("api/app.py", "api/: app.py routes.py", True),            # a listing line
    ("tokenmizer/agents/__init__.py", "tokenmizer/agents/: __init__.py", True),
    ("api/app.py", "api/: routes.py\napp.py alone", False),    # name on another line
    ("api/app.py", "other/: app.py", False),                   # wrong directory
    ("api/app.py", "nothing here", False),
    ("plainword", "a plainword in text", True),
    ("plainword", "different", False),
    ("a/b.py", "x/a/: b.py", True),
])
def test_present(token, text, expected):
    assert present(token, text) is expected


# ── What the agent's next step needed ────────────────────────────────────────

def _history(tool_text: str, user_text: str = "go") -> list[dict]:
    return [{"role": "user", "content": user_text},
            {"role": "tool", "tool_call_id": "c1", "content": tool_text}]


def test_needed_is_what_the_next_action_took_from_earlier_output():
    history = _history("found src/handler/route_table.py in the listing")
    action = '{"file_path": "src/handler/route_table.py"}'
    assert "src/handler/route_table.py" in needed_from_history(action, history)


def test_something_the_history_never_showed_is_not_a_need():
    """The agent made it up or knew it already; the proxy cannot have lost it."""
    assert needed_from_history('{"file_path": "invented/thing.py"}',
                               _history("nothing relevant")) == set()


def test_ordinary_tokens_are_not_information():
    """A word that occurs many times (a JSON key, "command") is not something
    the proxy could be blamed for dropping."""
    history = _history("command " * 60)
    assert needed_from_history('{"command": "ls"}', history) == set()


def test_short_tokens_are_ignored():
    assert needed_from_history('{"a": "ab"}', _history("ab ab")) == set()


def test_only_tool_output_and_user_turns_count_as_the_source():
    """What the assistant said itself is not something it learned."""
    history = [{"role": "assistant", "content": "I will edit only/said/by_me.py"}]
    assert needed_from_history('{"p": "only/said/by_me.py"}', history) == set()


# ── The count ────────────────────────────────────────────────────────────────

def test_retention_counts_what_survived_and_lists_what_did_not():
    requests = [_history("see src/kept/one.py and src/lost/two.py")]
    actions = ['{"a": "src/kept/one.py", "b": "src/lost/two.py"}']
    needed, kept, lost = retention(requests, actions, ["only src/kept/one.py was sent"])
    assert (needed, kept) == (2, 1)
    assert lost == ["src/lost/two.py"]


def test_retention_of_everything_sent_is_total():
    requests = [_history("see src/kept/one.py")]
    actions = ['{"a": "src/kept/one.py"}']
    needed, kept, lost = retention(requests, actions, ["src/kept/one.py"])
    assert needed == kept == 1 and lost == []


# ── Loading a transcript ─────────────────────────────────────────────────────

def _line(role, content, **extra):
    import json
    return json.dumps({"type": role, "message": {"role": role, "content": content}, **extra})


def test_a_transcript_becomes_requests_with_the_action_that_followed(tmp_path):
    lines = [
        _line("user", "fix the bug in api/app.py"),
        _line("assistant", [{"type": "tool_use", "id": "c1", "name": "read",
                             "input": {"path": "api/app.py"}}]),
        _line("user", [{"type": "tool_result", "tool_use_id": "c1", "content": "def f(): ..."}]),
        _line("assistant", [{"type": "text", "text": "done"}]),
    ]
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    requests, actions = load_trace_with_actions(str(path), 10)

    assert len(requests) == len(actions) == 2
    assert requests[0] == [{"role": "user", "content": "fix the bug in api/app.py"}]
    assert "api/app.py" in actions[0], "the action is what the agent did next"
    assert requests[1][-1] == {"role": "tool", "tool_call_id": "c1", "content": "def f(): ..."}
    assert any(m.get("tool_calls") for m in requests[1])


def test_sidechains_and_bookkeeping_lines_are_left_out(tmp_path):
    """A sub-agent's turns are not part of the conversation the client sends.
    The side question is what would show up in the history if the filter
    failed; a side reply alone would make the same request either way."""
    lines = [
        _line("user", "hello"),
        _line("user", "a question asked inside a sub-agent", isSidechain=True),
        '{"type": "summary", "summary": "not a message"}',
        "not json at all",
        _line("assistant", "hi"),
    ]
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    requests, _ = load_trace_with_actions(str(path), 10)
    assert requests == [[{"role": "user", "content": "hello"}]]
