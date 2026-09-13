"""
LangChain integration — a BaseChatMessageHistory backed by TokenMizer's
graph memory.

Behaves exactly like LangChain's own InMemoryChatMessageHistory for the
standard RunnableWithMessageHistory contract: `messages` returns every
turn verbatim, `add_messages` appends, `clear` empties it. On top of
that, every added message is also run through tokenmizer.agents.Memory,
so you additionally get long-horizon graph memory — context(), search(),
decisions(), errors(), why() — for the same session_id, surviving process
restarts, with no second store to wire up.

Requires langchain-core (not a TokenMizer dependency — install it
yourself: `pip install langchain-core`).
"""
from __future__ import annotations

from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

from tokenmizer.agents import Memory

# LangChain's message "type" strings -> the {"role": ...} TokenMizer's
# extractor expects. Anything else (tool, function, ...) is skipped: it
# carries no prose for the heuristic extractor to read, and passing an
# unrecognized role through unchanged would look like a real user/
# assistant turn to it.
_ROLE_FOR_TYPE = {"human": "user", "ai": "assistant", "system": "system"}


def _to_plain(message: BaseMessage) -> dict | None:
    if message.type not in _ROLE_FOR_TYPE or not isinstance(message.content, str):
        return None
    return {"role": _ROLE_FOR_TYPE[message.type], "content": message.content}


class TokenMizerChatMessageHistory(BaseChatMessageHistory):
    """Drop-in for `RunnableWithMessageHistory`'s `get_session_history`,
    with graph memory attached.

    `session_id` is TokenMizer's session — pass the same value your
    `get_session_history` factory receives as its key, so the graph
    memory and the message buffer stay the same session.
    """

    def __init__(self, session_id: str, storage_dir: str = "./checkpoints"):
        self.session_id = session_id
        self._messages: list[BaseMessage] = []
        self._memory = Memory(session_id, storage_dir=storage_dir)

    @property
    def messages(self) -> list[BaseMessage]:
        return list(self._messages)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        self._messages.extend(messages)
        plain = [p for m in messages if (p := _to_plain(m)) is not None]
        if plain:
            self._memory.add(plain)

    def clear(self) -> None:
        self._messages = []

    # TokenMizer-specific — not part of BaseChatMessageHistory's contract,
    # but why you would reach for this class over LangChain's own buffer.
    def context(self, token_budget: int = 400) -> str:
        """A resume block — goal, open work, decisions, files, open
        errors — to inject into a system prompt for recall a verbatim
        message buffer alone does not give you (e.g. after the buffer
        itself has been trimmed for length)."""
        return self._memory.context(token_budget=token_budget)

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        """One specific fact, ranked by relevance, instead of scanning
        every verbatim message for it."""
        return self._memory.search(query, top_k=top_k)


def main() -> None:
    import tempfile

    from langchain_core.messages import AIMessage, HumanMessage

    with tempfile.TemporaryDirectory() as tmp:
        history = TokenMizerChatMessageHistory("demo-langchain", storage_dir=tmp)
        history.add_messages([
            HumanMessage("We need to build the checkout service. What database should we use?"),
            AIMessage("Decided: use PostgreSQL for order storage."),
        ])
        print(f"Verbatim messages: {len(history.messages)}")
        print(f"Graph memory context:\n{history.context()}")


if __name__ == "__main__":
    main()
