"""
The one typed object that crosses a layer boundary.

This module used to hold thirteen DTOs and a docstring declaring "no raw
dict crosses a layer boundary". Twelve of them were never imported
anywhere, by any layer, and the code they described passes dicts — so the
rule was a statement of intent that the codebase had quietly stopped
following, and the file read as an architecture nobody could find.

Deleted rather than wired up: a typed boundary is worth having where a
shape is easy to get wrong, and the surviving one is that case —
`GraphMemory.get_stats()` is read by the CLI, the MCP server and
`/api/stats`, which is exactly where an un-named dict goes stale. Adding
the other twelve back is a change to make WITH the callers that need
them, not before.

tokenmizer/core/dto.py
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphStatsDTO:
    session_id: str
    node_count: int
    edge_count: int
    by_type: dict
    by_status: dict
    processed_messages: int
    avg_confidence: float
    # Non-zero means decision-transition tracking (the "Changed X → Y"
    # line in resume context) is degraded, even though node creation
    # itself is still working.
    decision_tracking_failures: int = 0
    # True if SQLite could not be initialized for this session — the graph
    # is running in-memory only, with NO durable persistence whatsoever.
    # A restart will lose everything.
    persistence_broken: bool = False
    # True if stored memory was destroyed during corruption recovery
    # (this session's row dropped, or the shared DB file quarantined).
    # persistence_broken is about future writes; this is about past data
    # that no longer exists. Without it, a health check watching
    # node_count cannot tell "new session" from "we lost your memory".
    data_loss_detected: bool = False
    # True when the last SQLite read failed (lock contention, mostly) and
    # this instance holds an empty graph that does not reflect what is
    # stored. persist() refuses to write while it is set, so nothing is
    # lost — but a node_count of 0 under this flag is not the truth, and
    # a page rendering it must say "could not read", not "empty".
    load_failed: bool = False
