"""
In-process memory for agents.

    from tokenmizer.agents import Memory

    memory = Memory("order-service")
    memory.add(messages)                      # a list of {"role", "content"}
    memory.search("what are we storing orders in")
    memory.context()                          # a resume block for a system prompt
    memory.why("postgres")                    # the decision trail

Everything here already existed as GraphMemory, the reasoning module and the
checkpoint manager. What did not exist was a way to use them from an agent
without running the HTTP proxy or learning the graph's internals: the
package's own README compared itself to Mem0 and Zep, whose whole interface
is `add` and `search`, while this package's equivalent was
`extract_from_messages` and a method that returns node dataclasses. This
module is that interface — a thin facade, no new storage and no new
extraction. It works with no server, no API key and no network.

Session isolation is by `session_id`; storage is the same SQLite file the
proxy uses, so an agent and the proxy can share a session.
"""
from __future__ import annotations

from typing import Iterable, Optional

from tokenmizer.graph_memory.graph import GraphMemory
from tokenmizer.graph_memory.types import NodeType

__all__ = ["Memory"]


class Memory:
    """Graph memory for one session, usable from any agent loop."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        storage_dir: str = "./checkpoints",
        semantic_retrieval: bool = False,
    ):
        if not session_id:
            from tokenmizer.core.session import default_session_id
            session_id = default_session_id()
            if session_id is None:
                raise ValueError(
                    "No session_id given and the working directory has no "
                    "usable name to derive one from; pass session_id explicitly."
                )
        self.session_id = session_id
        self._graph = GraphMemory(
            session_id, storage_dir=storage_dir,
            semantic_retrieval=semantic_retrieval,
        )

    # ── Write ────────────────────────────────────────────────────────────────

    def add(self, messages: Iterable[dict]) -> int:
        """Extract from `messages` ({"role", "content"} each) into memory.

        Returns the number of nodes the graph gained. Idempotent for
        messages already seen — the graph tracks processed content, so
        passing a growing conversation each turn is the intended use.
        """
        before = len(self._graph._nodes)
        self._graph.extract_from_messages(list(messages), incremental=True)
        return len(self._graph._nodes) - before

    def add_text(self, text: str, role: str = "assistant") -> int:
        """Convenience for a single message."""
        return self.add([{"role": role, "content": text}])

    # ── Read ─────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 8,
               types: Optional[Iterable[str]] = None) -> list[dict]:
        """Ranked memories relevant to `query`, as plain dicts.

        `types` filters to node types by name ("decision", "error", "task",
        "file", "goal", ...). Each result carries the fields an agent can
        act on; the graph's internal ids stay internal.
        """
        wanted = {NodeType(t) for t in types} if types else None
        results = []
        for node in self._graph.query(query, top_k=top_k * 2 if wanted else top_k):
            if wanted and node.type not in wanted:
                continue
            results.append(self._to_dict(node))
            if len(results) >= top_k:
                break
        return results

    def decisions(self) -> list[dict]:
        """Every active decision, most important first."""
        return self._by_type(NodeType.DECISION)

    def errors(self) -> list[dict]:
        """Every recorded failure, most important first."""
        return self._by_type(NodeType.ERROR)

    def context(self, token_budget: int = 400) -> str:
        """A resume block sized for a system prompt: goal, open work, decisions,
        files, open errors — the same block the proxy injects."""
        return self._graph.to_context_block(token_budget=token_budget)

    def why(self, query: str) -> dict:
        """The trail behind a decision: what it replaced, and why."""
        from tokenmizer.graph_memory.reasoning import why
        return why(self._graph, query)

    def impact(self, query: str) -> dict:
        """What is connected to a memory — the files a decision touches, the
        tasks an error blocks."""
        from tokenmizer.graph_memory.reasoning import impact
        return impact(self._graph, query)

    def stats(self) -> dict:
        return self._graph.stats()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def save(self) -> bool:
        """Force a write. add() persists on its own; this is for shutdown
        paths that want to be certain."""
        return self._graph._persist(force=True)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _by_type(self, node_type: NodeType) -> list[dict]:
        from tokenmizer.graph_memory.graph import INACTIVE_STATUSES
        nodes = [
            n for n in self._graph._nodes.values()
            if n.type == node_type and not n._evicted
            and n.status not in INACTIVE_STATUSES
        ]
        nodes.sort(key=lambda n: n.importance, reverse=True)
        return [self._to_dict(n) for n in nodes]

    @staticmethod
    def _to_dict(node) -> dict:
        return {
            "type": node.type.value,
            "label": node.label,
            "summary": node.summary,
            "status": node.status.value,
            "confidence": node.confidence,
            "importance": node.importance,
            "created_at": node.created_at,
        }
