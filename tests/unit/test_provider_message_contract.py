"""
Cross-layer contract: whatever the pipeline hands a provider must be a
legal input for that provider.

Every other test in this suite checks one layer against its own contract.
Nothing checked the seam, and a two-line slice in SmartMessageWindow shipped
an assistant-first message list to Anthropic — which rejects it with a 400
"first message must use the 'user' role". Because windowing only engages once
a session passes max_tokens_before_summary, and a session never shrinks back
under it, that turned into every subsequent turn of a long session failing.

The rule these tests lock down: for providers that hoist system messages out
of the conversation (Anthropic, Gemini), the remaining conversation must
begin with a user turn.
"""
from __future__ import annotations

import pytest

from tokenmizer.compression.window import SmartMessageWindow
from tokenmizer.providers.providers import conversation_messages


class _StubGraph:
    def to_context_block(self, token_budget: int = 250) -> str:
        return "Goal: ship auth\nDecided: PostgreSQL | bcrypt"


def _session(turns: int) -> list[dict]:
    """A realistic chat request: system prompt, N full turns, ending on user."""
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(turns):
        msgs.append({"role": "user", "content": f"user turn {i} " + "word " * 60})
        msgs.append({"role": "assistant", "content": f"assistant turn {i} " + "word " * 60})
    msgs.append({"role": "user", "content": "what did we decide about the database?"})
    return msgs


# ── The bug itself ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("protect_recent", [2, 4, 6, 8, 10, 12])
def test_windowed_conversation_starts_with_user(protect_recent):
    """An even protect_recent used to land the slice on an assistant turn.

    Parametrised over both parities because the failure was parity-dependent:
    odd values happened to work, and the default (10) did not.
    """
    windowed, _ = SmartMessageWindow(
        token_budget=1000, protect_recent=protect_recent
    ).apply(_session(30), _StubGraph(), "gpt-4o")

    conv = [m for m in windowed if m.get("role") != "system"]
    assert conv, "windowing must not empty the conversation"
    assert conv[0]["role"] == "user", (
        f"protect_recent={protect_recent} produced a {conv[0]['role']}-first "
        "conversation; Anthropic and Gemini reject that with a 400"
    )


def test_windowing_preserves_the_final_user_turn():
    """Trimming the leading turn must never eat the question being asked."""
    session = _session(30)
    windowed, _ = SmartMessageWindow(token_budget=1000, protect_recent=10).apply(
        session, _StubGraph(), "gpt-4o"
    )
    assert windowed[-1]["content"] == session[-1]["content"]


def test_windowing_still_saves_tokens():
    """The fix drops at most one message — it must not cost the whole saving."""
    _, saved = SmartMessageWindow(token_budget=1000, protect_recent=10).apply(
        _session(30), _StubGraph(), "gpt-4o"
    )
    assert saved > 0


# ── The guard, so no future layer can re-break it ────────────────────────────

def test_conversation_messages_drops_leading_assistant():
    conv = conversation_messages([
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "orphaned reply"},
        {"role": "user", "content": "question"},
    ])
    assert [m["role"] for m in conv] == ["user"]


def test_conversation_messages_keeps_a_valid_conversation_intact():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    assert conversation_messages(msgs) == msgs[1:]


def test_conversation_messages_on_assistant_only_input():
    """No user turn at all: return empty rather than an illegal payload.

    The caller raises a clear error; sending it would be a provider 400 with
    no indication of which layer produced it.
    """
    assert conversation_messages([{"role": "assistant", "content": "x"}]) == []
