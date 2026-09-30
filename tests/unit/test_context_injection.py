"""
Regression test for TM-10: graph context injection silently no-ops when
the outgoing message list has no system message.

Bug: the injection block found a system message's index and mutated it
IF one existed, with no else branch — so a request with no system
message did the graph query, built the context block, and threw it
away. A system message was only guaranteed to exist because layer 2
(terse-output injection) adds one, and only when
settings.terse_output.enabled is True — a completely unrelated setting.
Turning that off silently disabled graph context injection too.
"""
from __future__ import annotations

import pytest

from tokenmizer.api import app as app_module
from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType


@pytest.fixture
def graph_with_signal(tmp_path):
    g = GraphMemory("ctx-inject-test", storage_dir=str(tmp_path))
    g.add_node(NodeType.DECISION, "Use PostgreSQL for the primary datastore",
              NodeStatus.COMPLETED, summary="relational integrity matters here",
              importance=0.9)
    g.add_node(NodeType.TASK, "Implement the user authentication flow",
              NodeStatus.IN_PROGRESS, importance=0.8)
    g.add_node(NodeType.FILE, "api/auth.py", NodeStatus.IN_PROGRESS, importance=0.7)
    return g


QUESTION = "how are we handling database access right now"


def _last_user(messages):
    return next(m for m in reversed(messages) if m.get("role") == "user")


class TestContextInjectionWithoutExistingSystemMessage:

    async def test_context_is_injected_even_with_no_system_message(
        self, graph_with_signal, monkeypatch
    ):
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "enabled", False)
        raw = [{"role": "user", "content": QUESTION}]
        messages = [dict(m) for m in raw]

        updated, _ = await app_module._update_graph(
            "ctx-inject-test", graph_with_signal, raw, messages,
            "claude-sonnet-4-6", {}, raw[0]["content"],
        )

        turn = _last_user(updated)["content"].lower()
        assert "relevant session context" in turn, (
            "relevant graph context was found but is not in the request"
        )
        assert "postgresql" in turn or "auth" in turn

    async def test_the_question_itself_is_unchanged_and_comes_first(
        self, graph_with_signal, monkeypatch
    ):
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "enabled", False)
        raw = [{"role": "user", "content": QUESTION}]
        messages = [dict(m) for m in raw]

        updated, _ = await app_module._update_graph(
            "ctx-inject-test", graph_with_signal, raw, messages,
            "claude-sonnet-4-6", {}, raw[0]["content"],
        )
        assert _last_user(updated)["content"].startswith(QUESTION)
        assert raw[0]["content"] == QUESTION, "the client's own message was mutated"

    async def test_existing_system_message_is_left_exactly_as_sent(
        self, graph_with_signal, monkeypatch
    ):
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "enabled", False)
        raw = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": QUESTION},
        ]
        messages = [dict(m) for m in raw]

        updated, _ = await app_module._update_graph(
            "ctx-inject-test", graph_with_signal, raw, messages,
            "claude-sonnet-4-6", {}, raw[1]["content"],
        )
        system_msgs = [m for m in updated if m.get("role") == "system"]
        assert len(system_msgs) == 1, "must not create a SECOND system message"
        assert system_msgs[0]["content"] == "You are a helpful coding assistant."


class TestInjectedContextKeepsTheCacheablePrefixStable:
    """Providers cache the longest byte-identical prefix of a request and
    render the system prompt before the conversation. A block that changes
    every turn must therefore sit after both, or every request re-pays for
    the whole history."""

    async def test_system_prompt_and_earlier_turns_go_out_unchanged(
        self, graph_with_signal, monkeypatch
    ):
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "enabled", False)
        stable = "You are the deployment assistant. " * 40
        raw = [
            {"role": "user", "content": "we picked a datastore earlier"},
            {"role": "assistant", "content": "Yes, noted."},
            {"role": "user", "content": QUESTION},
        ]
        messages = [{"role": "system", "content": stable}] + [dict(m) for m in raw]

        updated, _ = await app_module._update_graph(
            "ctx-inject-test", graph_with_signal, raw, messages,
            "claude-sonnet-4-6", {}, QUESTION,
        )
        assert updated[0] == {"role": "system", "content": stable}
        assert updated[1:3] == raw[:2]
        assert "relevant session context" in updated[-1]["content"]

    async def test_a_trailing_tool_result_keeps_the_system_placement(
        self, graph_with_signal, monkeypatch
    ):
        """Only a user turn is a safe place to append to; a tool result
        closing an agent step is left as the provider expects it."""
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "enabled", False)
        raw = [
            {"role": "user", "content": QUESTION},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
        messages = [dict(m) for m in raw]

        updated, _ = await app_module._update_graph(
            "ctx-inject-test", graph_with_signal, raw, messages,
            "claude-sonnet-4-6", {}, QUESTION,
        )
        assert updated[-1] == raw[-1]
        system = next(m["content"] for m in updated if m["role"] == "system")
        assert "relevant session context" in system


class TestContextTheRequestAlreadyCarriesIsNotRepeated:

    async def test_a_short_session_gets_no_duplicate_of_its_own_history(
        self, graph_with_signal, monkeypatch
    ):
        """Every node was extracted from a turn still in this request, so
        the block would only repeat it. This was 81% extra input on a
        ten-turn session."""
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "enabled", False)
        raw = [
            {"role": "user", "content": "Use PostgreSQL for the primary datastore "
                                        "because relational integrity matters here"},
            {"role": "assistant", "content": "Agreed."},
            {"role": "user", "content": "Implement the user authentication flow "
                                        "in api/auth.py next"},
            {"role": "assistant", "content": "On it."},
            {"role": "user", "content": QUESTION},
        ]
        messages = [dict(m) for m in raw]

        updated, _ = await app_module._update_graph(
            "ctx-inject-test", graph_with_signal, raw, messages,
            "claude-sonnet-4-6", {}, QUESTION,
        )
        assert updated == raw

    async def test_a_fact_whose_turn_is_gone_is_still_injected(
        self, graph_with_signal, monkeypatch
    ):
        """The same graph, with the turns that stated it no longer in the
        request (windowed, or truncated by the client) — the reason the
        block exists."""
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "enabled", False)
        raw = [{"role": "user", "content": QUESTION}]
        messages = [dict(m) for m in raw]

        updated, _ = await app_module._update_graph(
            "ctx-inject-test", graph_with_signal, raw, messages,
            "claude-sonnet-4-6", {}, QUESTION,
        )
        assert "postgresql" in updated[-1]["content"].lower()

    async def test_the_payload_word_sets_are_built_off_the_event_loop(
        self, graph_with_signal, monkeypatch
    ):
        """Building a word set for every message in the payload is linear in
        the payload, which on an agent session is tool output: it runs in a
        worker thread so the event loop keeps serving other requests."""
        import threading
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "enabled", False)
        loop_thread = threading.get_ident()
        threads = []
        real = app_module._content_words
        # A node's own label (a few words) is fine on the loop; the payload's
        # messages are what grows, so only those are tracked.
        monkeypatch.setattr(app_module, "_content_words", lambda t: (
            threads.append(threading.get_ident()) if QUESTION in t else None, real(t))[1])
        raw = [{"role": "user", "content": QUESTION}]
        await app_module._update_graph(
            "ctx-inject-test", graph_with_signal, raw, [dict(m) for m in raw],
            "claude-sonnet-4-6", {}, QUESTION,
        )
        assert threads and loop_thread not in threads

    def test_a_paraphrase_is_not_mistaken_for_the_fact(self):
        """Conservative by design: if one word of the label is missing the
        node is treated as absent, and injected."""
        node = type("N", (), {"label": "Use PostgreSQL for sessions", "summary": ""})()
        words = [app_module._content_words("we store sessions in postgres")]
        assert not app_module._already_in_payload(node, words)

    def test_a_file_named_with_and_without_its_extension_is_the_same_file(self):
        """A node label is clipped to a clause and can lose the extension the
        turn spelled out. Reading `hybrid_extractor` as a different word from
        `hybrid_extractor.py` sent a fenced block, boilerplate included, to
        repeat a sentence the model was already reading, on every later turn
        of a 20-turn session."""
        node = type("N", (), {
            "label": "Replaced the pattern with typed-exception matching in "
                     "hybrid_extractor", "summary": ""})()
        said = ("Replaced the pattern with typed-exception matching in "
                "hybrid_extractor.py. Error F1 went from 14 to 87 percent.")
        assert app_module._already_in_payload(node, [app_module._content_words(said)])

    def test_a_different_extension_is_a_different_file(self):
        """The stem match must not turn one file into another: a label that
        names config.py is not stated by a turn that only says config.yaml."""
        node = type("N", (), {"label": "Edit config.py", "summary": ""})()
        words = [app_module._content_words("Edit config.yaml")]
        assert not app_module._already_in_payload(node, words)

    def test_a_sentence_ending_in_a_stem_needs_the_stem_in_the_turn(self):
        node = type("N", (), {"label": "Fix hybrid_extractor", "summary": ""})()
        words = [app_module._content_words("Fix hybrid_extractors instead")]
        assert not app_module._already_in_payload(node, words)

    def test_words_split_across_two_messages_do_not_count(self):
        node = type("N", (), {"label": "Use PostgreSQL for sessions", "summary": ""})()
        words = [app_module._content_words("PostgreSQL it is"),
                 app_module._content_words("sessions next")]
        assert not app_module._already_in_payload(node, words)
