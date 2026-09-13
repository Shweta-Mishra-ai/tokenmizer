"""
LangGraph integration — a checkpointer that also feeds a graph run's
messages into TokenMizer's graph memory.

Checkpoint durability itself is delegated to LangGraph's own
InMemorySaver: LangGraph's checkpoint format (channel values, pending
writes, versions) is an implementation detail this adapter has no reason
to reimplement, and getting it subtly wrong would break graph replay in
ways a small example is the wrong place to risk. What this class adds is
a side channel — every put() that carries a "messages" channel (the
convention `langgraph.graph.MessagesState` and most chat-style graphs
use) also runs those messages through a tokenmizer.agents.Memory keyed by
the run's thread_id, so `context()`/`search()`/`why()` are available for
whatever conversation the graph is having, with no second store to wire
up. Swap `InMemorySaver` for a durable LangGraph checkpointer (Postgres,
SQLite) in production; this adapter works the same way layered on any of
them.

Requires langgraph and langchain-core (not TokenMizer dependencies —
install them yourself: `pip install langgraph langchain-core`).
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

from tokenmizer.agents import Memory

logger = logging.getLogger(__name__)

_ROLE_FOR_TYPE = {"human": "user", "ai": "assistant", "system": "system"}


class TokenMizerCheckpointSaver(InMemorySaver):
    """An InMemorySaver that also extracts graph memory per thread_id."""

    def __init__(self, *, storage_dir: str = "./checkpoints"):
        super().__init__()
        self._storage_dir = storage_dir
        self._memory_by_thread: dict[str, Memory] = {}

    def memory_for(self, thread_id: str) -> Memory:
        """The tokenmizer.agents.Memory for one LangGraph thread — call
        this directly for context()/search()/decisions()/errors()/why()
        instead of going through the checkpointer's own read path, which
        only knows how to replay LangGraph state."""
        if thread_id not in self._memory_by_thread:
            self._memory_by_thread[thread_id] = Memory(thread_id, storage_dir=self._storage_dir)
        return self._memory_by_thread[thread_id]

    def put(self, config, checkpoint, metadata, new_versions):
        result = super().put(config, checkpoint, metadata, new_versions)
        self._extract(config, checkpoint)
        return result

    async def aput(self, config, checkpoint, metadata, new_versions):
        result = await super().aput(config, checkpoint, metadata, new_versions)
        self._extract(config, checkpoint)
        return result

    def _extract(self, config: dict, checkpoint: dict) -> None:
        thread_id = (config.get("configurable") or {}).get("thread_id")
        messages = (checkpoint.get("channel_values") or {}).get("messages")
        if not thread_id or not messages:
            return
        plain = [
            {"role": _ROLE_FOR_TYPE[m.type], "content": m.content}
            for m in messages
            if isinstance(m, BaseMessage) and m.type in _ROLE_FOR_TYPE
            and isinstance(m.content, str)
        ]
        if not plain:
            return
        try:
            self.memory_for(thread_id).add(plain)
        except Exception as e:
            # Extraction is a side channel; the checkpoint itself already
            # committed above, and a graph run must not fail because its
            # bonus graph-memory extraction did.
            logger.warning(f"TokenMizer extraction failed for thread {thread_id!r}: {e}")


def main() -> None:
    import tempfile

    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.graph import END, START, MessagesState, StateGraph

    def call_model(state: MessagesState) -> dict[str, Any]:
        # Swap this for a real provider call.
        return {"messages": [AIMessage("Decided: use PostgreSQL for order storage.")]}

    with tempfile.TemporaryDirectory() as tmp:
        checkpointer = TokenMizerCheckpointSaver(storage_dir=tmp)
        graph = StateGraph(MessagesState)
        graph.add_node("call_model", call_model)
        graph.add_edge(START, "call_model")
        graph.add_edge("call_model", END)
        app = graph.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "demo-langgraph"}}
        app.invoke(
            {"messages": [HumanMessage(
                "We need to build the checkout service. What database should we use?"
            )]},
            config=config,
        )
        print(f"Graph memory context:\n{checkpointer.memory_for('demo-langgraph').context()}")


if __name__ == "__main__":
    main()
