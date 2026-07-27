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


class TestContextInjectionWithoutExistingSystemMessage:

    async def test_context_is_injected_even_with_no_system_message(
        self, graph_with_signal, monkeypatch
    ):
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "enabled", False)
        raw = [{"role": "user", "content": "how are we handling database access right now"}]
        messages = [dict(m) for m in raw]

        updated, _ = await app_module._update_graph(
            "ctx-inject-test", graph_with_signal, raw, messages,
            "claude-sonnet-4-6", {}, raw[0]["content"],
        )

        system_msgs = [m for m in updated if m.get("role") == "system"]
        assert system_msgs, (
            "no system message exists in the request, but relevant graph "
            "context was found — it must be injected as a NEW system "
            "message, not silently discarded"
        )
        assert "postgresql" in system_msgs[0]["content"].lower() or \
               "auth" in system_msgs[0]["content"].lower(), (
            f"injected system message doesn't contain the expected graph "
            f"context: {system_msgs[0]['content']!r}"
        )

    async def test_injected_system_message_is_first_in_list(
        self, graph_with_signal, monkeypatch
    ):
        """Convention: system message goes first, matching how it's
        placed everywhere else in this codebase (compression layer,
        smart window bridge message, etc.)."""
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "enabled", False)
        raw = [{"role": "user", "content": "how are we handling database access right now"}]
        messages = [dict(m) for m in raw]

        updated, _ = await app_module._update_graph(
            "ctx-inject-test", graph_with_signal, raw, messages,
            "claude-sonnet-4-6", {}, raw[0]["content"],
        )
        assert updated[0]["role"] == "system"

    async def test_existing_system_message_still_gets_context_prepended(
        self, graph_with_signal, monkeypatch
    ):
        """Regression guard: must not break the EXISTING behavior (a
        system message that's already present gets the context
        prepended to it) while fixing the missing-system-message case."""
        monkeypatch.setattr(app_module.settings.graph_checkpoint, "enabled", False)
        raw = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "how are we handling database access right now"},
        ]
        messages = [dict(m) for m in raw]

        updated, _ = await app_module._update_graph(
            "ctx-inject-test", graph_with_signal, raw, messages,
            "claude-sonnet-4-6", {}, raw[1]["content"],
        )
        system_msgs = [m for m in updated if m.get("role") == "system"]
        assert len(system_msgs) == 1, "must not create a SECOND system message"
        assert "You are a helpful coding assistant." in system_msgs[0]["content"]
