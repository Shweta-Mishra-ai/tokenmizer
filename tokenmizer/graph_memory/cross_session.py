"""
Cross-session recall: rank one principal's other sessions' nodes alongside
the current session's, each result paired with the session_id it came
from.

Gated by Settings.graph_checkpoint.cross_session_recall (default off) and
wired in from two call sites: tokenmizer.agents.Memory.search() and the
proxy's context-injection step in api/app.py. Both already know which
other session_ids belong to the same principal — security/ownership.py is
the existing sessions-belong-to-a-principal boundary, and this module
never widens it; it only reads across sessions the caller has already
established are the same principal's.

Off by default for the same reason semantic_retrieval is: merging in
another session's nodes changes what benchmarks.eval's recall/decision/
error fixtures see, and a change to graph_memory ranking that looks
correct by inspection has regressed those fixtures before (see project
memory) — it must be measured on a real multi-session corpus, not assumed
safe.
"""
from __future__ import annotations

import logging
from typing import Callable, Iterable

from tokenmizer.graph_memory.graph import GraphMemory
from tokenmizer.graph_memory.types import MemoryNode

logger = logging.getLogger(__name__)

# ponytail: bounds how many of a principal's other sessions get pulled into
# one query — a principal with hundreds of sessions would otherwise load
# that many GraphMemory instances (one SQLite read each) per call. Raise
# if measurement shows recall needs a deeper session history than this.
MAX_OTHER_SESSIONS = 5


def query_across_sessions(
    current: GraphMemory,
    other_session_ids: Iterable[str],
    task: str,
    top_k: int,
    load_graph: Callable[[str], GraphMemory],
) -> list[tuple[MemoryNode, str]]:
    """Rank `task` against `current` plus up to MAX_OTHER_SESSIONS other
    sessions, each result paired with the session_id it came from.

    Scored together (not top_k-per-session then merged) so a session with
    one very relevant node isn't crowded out by another session's many
    mediocre ones. `load_graph(session_id) -> GraphMemory` loads (or
    reuses a cached) graph for one other session — the caller supplies it
    so this function stays agnostic to whether that's the proxy's LRU
    cache or a fresh GraphMemory() for standalone Memory() use. A session
    that fails to load is skipped rather than failing the whole query —
    one unreadable session must not take down retrieval for the rest.
    """
    pool: list[tuple[float, MemoryNode, str]] = [
        (score, node, current.session_id)
        for score, node in current._score_nodes(task)
    ]

    for session_id in list(other_session_ids)[:MAX_OTHER_SESSIONS]:
        try:
            other = load_graph(session_id)
        except Exception as e:
            # One unreadable session must not sink the whole recall, but
            # it must not be invisible either: silently returning fewer
            # results looks exactly like "nothing relevant was found".
            logger.warning(
                "Cross-session recall skipped %s (its graph could not be "
                "loaded, so anything it knows is missing from this "
                "answer): %s", session_id, e,
            )
            continue
        for score, node in other._score_nodes(task):
            pool.append((score, node, session_id))

    pool.sort(key=lambda x: x[0], reverse=True)
    return [(node, session_id) for _, node, session_id in pool[:top_k]]
