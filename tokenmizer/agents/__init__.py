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

import logging
from typing import Iterable, Optional

from tokenmizer.graph_memory.graph import GraphMemory
from tokenmizer.graph_memory.types import NodeType

logger = logging.getLogger(__name__)

__all__ = ["Memory"]


class Memory:
    """Graph memory for one session, usable from any agent loop."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        storage_dir: str = "./checkpoints",
        semantic_retrieval: Optional[bool] = None,
        cross_session_recall: Optional[bool] = None,
        principal: Optional[str] = None,
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
        self._storage_dir = storage_dir
        if semantic_retrieval is None:
            # None means "whatever this deployment is configured for",
            # which is "auto" by default: on when the embedding model
            # actually loads. An in-process caller gets the same ranking
            # the proxy gets, rather than a quietly worse one.
            from tokenmizer.config.settings import (
                get_settings,
                resolve_semantic_retrieval,
            )
            semantic_retrieval = resolve_semantic_retrieval(
                get_settings().graph_checkpoint.semantic_retrieval)
        self._graph = GraphMemory(
            session_id, storage_dir=storage_dir,
            semantic_retrieval=semantic_retrieval,
        )
        if cross_session_recall is None:
            from tokenmizer.config.settings import get_settings
            cross_session_recall = get_settings().graph_checkpoint.cross_session_recall
        self._cross_session_recall = cross_session_recall
        # No API key exists at this layer (Memory works with no server, no
        # key, no network — see the module docstring), so there is no
        # credential to derive a principal from the way api/app.py does.
        # DEV_PRINCIPAL is security/ownership.py's own sentinel for exactly
        # this case: every caller with no credential is the same principal.
        # Passing an explicit `principal` is only useful when several
        # independent local callers share one storage_dir and must not see
        # each other's sessions in cross-session recall.
        if principal is None:
            from tokenmizer.security.ownership import DEV_PRINCIPAL
            principal = DEV_PRINCIPAL
        self._principal = principal
        # Claimed unconditionally, not gated by cross_session_recall: the
        # proxy claims ownership for every chat request the same way (see
        # api/app.py), and cross-session recall can only discover a
        # session that was claimed. Gating this on the setting would mean
        # a session created while the setting was off stays invisible
        # forever even after it's turned on. Best-effort — a store that
        # can't be written just means this session won't be discoverable
        # from another one; Memory has no request to fail closed on.
        try:
            from tokenmizer.security.ownership import OwnershipStore
            OwnershipStore(storage_dir=storage_dir).claim(session_id, principal)
        except Exception:
            pass

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
        act on, plus `session_id` — the session it was retrieved from,
        always the current one unless cross_session_recall is on (see
        __init__ and Settings.graph_checkpoint.cross_session_recall), in
        which case it may name one of this principal's other sessions.
        The graph's internal ids stay internal.
        """
        wanted = {NodeType(t) for t in types} if types else None
        pool_k = top_k * 2 if wanted else top_k
        if self._cross_session_recall:
            pairs = self._cross_session_pool(query, pool_k)
        else:
            pairs = [(n, self.session_id) for n in self._graph.query(query, top_k=pool_k)]
        results = []
        for node, session_id in pairs:
            if wanted and node.type not in wanted:
                continue
            d = self._to_dict(node)
            d["session_id"] = session_id
            results.append(d)
            if len(results) >= top_k:
                break
        return results

    def _cross_session_pool(self, query: str, top_k: int) -> list[tuple]:
        from tokenmizer.graph_memory.cross_session import query_across_sessions
        return query_across_sessions(
            self._graph, self._other_session_ids(), query, top_k,
            self._load_other_graph,
        )

    def _other_session_ids(self) -> list[str]:
        """Every OTHER session this principal owns, per security/ownership.py.

        Claims this session for self._principal if it is not yet owned by
        anyone — Memory has no HTTP-layer request to deny access on, so
        unlike api/app.py this never rejects; a session already claimed by
        a different principal (e.g. through the proxy) is simply left
        alone and contributes nothing here.
        """
        from tokenmizer.security.ownership import OwnershipStore

        store = OwnershipStore(storage_dir=self._storage_dir)
        # An unreadable ownership store means cross-session recall returns
        # nothing, which is indistinguishable from "this principal owns no
        # other sessions" unless it is said out loud.
        try:
            owner = store.claim(self.session_id, self._principal)
        except Exception as e:
            logger.warning(
                "Ownership store unreadable, so no other session of this "
                "principal can be recalled: %s", e,
            )
            return []
        if owner != self._principal:
            return []
        try:
            sessions = store.sessions_for(owner)
        except Exception as e:
            logger.warning(
                "Could not list this principal's sessions, so cross-session "
                "recall has nothing to draw on: %s", e,
            )
            return []
        return [sid for sid in sessions if sid != self.session_id]

    def _load_other_graph(self, session_id: str) -> GraphMemory:
        return GraphMemory(
            session_id, storage_dir=self._storage_dir,
            semantic_retrieval=self._graph.semantic_retrieval,
        )

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
