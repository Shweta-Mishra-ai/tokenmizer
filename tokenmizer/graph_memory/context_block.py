"""
Resume context block — the tiered, token-budgeted summary injected into
the LLM at the start of a resumed session.

Extracted from graph.py to keep that file focused on core memory logic
(node/edge CRUD, extraction application, query, persistence). Follows
the same split pattern already established for visualization.py,
pruning.py, and persistence.py: a module-level function taking
`graph: GraphMemory` as its argument, with GraphMemory keeping a
one-line delegating method so existing callers are unaffected.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from tokenmizer.graph_memory.types import EdgeType, NodeStatus, NodeType

if TYPE_CHECKING:
    from tokenmizer.graph_memory.graph import GraphMemory


_DEFAULT_WORDS = {
    "goal": "Goal", "wip": "Working on", "done": "Done",
    "decided": "Decided", "files": "Files", "errors": "Open issues",
}


def _vocabulary(graph: "GraphMemory") -> dict:
    """The section headers for this graph's domain pack."""
    from tokenmizer.graph_memory.domains import get_pack

    domain = getattr(graph, "_domain", None)
    if domain is None:
        try:
            from tokenmizer.config.settings import get_settings
            domain = get_settings().domain
        except Exception:
            domain = None
    return {**_DEFAULT_WORDS, **get_pack(domain).vocabulary}


def to_context_block(graph: "GraphMemory", token_budget: int = 400) -> str:
    """
    Build tiered resume context block for LLM injection.

    Priority order (truncates from bottom if over budget):
      1. Goal                    — always shown (anchor)
      2. In-progress tasks       — sorted by importance (current focus)
      3. Recent completed tasks  — top 5 by recency, not all 50
      4. Active decisions        — top 6 by importance, with rationale
      5. Recent decision changes — transition summary (not strikethrough waste)
      6. Pending tasks           — what's next
      7. Files touched           — context for file-specific questions
      8. Environment             — versions, if present
      9. Open errors             — unresolved failures

    Quality rules applied:
    - SUPERSEDED decisions: shown only as "Changed X → Y" one-liner
      (not full label — wastes tokens showing wrong answer)
    - Completed tasks: importance-weighted, capped at 5 most recent
      (full history is in SQLite, not needed in resume)
    - Similar nodes: deduplicated by normalized label prefix
    - Transitions: shown as compact lines, not repeated decision labels
    """
    # (header, items, separator, priority) in display order; _pack decides
    # which items fit. See _pack for why this is per item, not per section.
    sections: list[tuple[str, list[str], str, int]] = []

    def _add(header: str, items: list[str], sep: str, prio: int) -> None:
        items = [i for i in items if i]
        if items:
            sections.append((header, items, sep, prio))
    # The pack renames the sections without changing what goes in them:
    # a research session's finished work is "Found", an incident's is
    # still "Done". Unknown keys fall back to the coding word, so a pack
    # that names nothing reads exactly as before.
    words = _vocabulary(graph)

    # ── 1. Goal ──────────────────────────────────────────────────────────
    goals = sorted(
        [n for n in graph._nodes.values()
         if n.type == NodeType.GOAL and not n._evicted],
        key=lambda x: x.importance, reverse=True
    )
    if goals:
        _add(words["goal"], [g.label for g in goals[:2]], " | ", prio=0)

    # ── 2. In-progress tasks ──────────────────────────────────────────────
    #
    # Most recently mentioned first. These were sorted by importance, which
    # every extracted task shares (0.6), so the order fell back to insertion
    # order — OLDEST first — and a long session's resume said it was
    # "working on" whatever it started 500 messages ago, never what it is
    # doing now. Ties within one extraction keep the later mention first.
    order = {nid: i for i, nid in enumerate(graph._nodes)}

    def _recent_first(n):
        return (n.importance, n.updated_at, order.get(n.id, 0))

    open_tasks = sorted(
        [n for n in graph._nodes.values()
         if n.type == NodeType.TASK
         and n.status == NodeStatus.IN_PROGRESS
         and not n._evicted],
        key=_recent_first, reverse=True
    )
    # ── 3. Pending tasks (next steps) ─────────────────────────────────────
    pending_tasks = sorted(
        [n for n in graph._nodes.values()
         if n.type == NodeType.TASK
         and n.status == NodeStatus.PENDING
         and not n._evicted],
        key=_recent_first, reverse=True
    )
    # Interleaved, so a tight budget keeps both the current step and the
    # next one rather than four in-progress items and no plan.
    current_work = [t for pair in zip(open_tasks[:4], pending_tasks[:4]) for t in pair]
    n = len(current_work) // 2
    current_work += open_tasks[n:4] + pending_tasks[n:4]
    if current_work:
        _add(words["wip"], [t.label for t in current_work], " | ", prio=2)

    # ── 4. Recent completed tasks — top 5 by recency+importance ───────────
    done = sorted(
        [n for n in graph._nodes.values()
         if n.type == NodeType.TASK
         and n.status == NodeStatus.COMPLETED
         and not n._evicted],
        key=lambda x: (x.updated_at * 0.6 + x.importance * 0.4),
        reverse=True
    )
    # Deduplicate: skip if label is very similar to already-included task
    done_deduped = []
    seen_prefixes: set[str] = set()
    for t in done:
        prefix = graph._normalize_label(t.label)[:20]
        if prefix not in seen_prefixes:
            done_deduped.append(t)
            seen_prefixes.add(prefix)
        if len(done_deduped) >= 6:
            break
    if done_deduped:
        _add(words["done"], [t.label for t in done_deduped], " | ", prio=5)

    # ── 5. Active decisions — top 6 by importance ─────────────────────────
    decisions = sorted(
        [n for n in graph._nodes.values()
         if n.type == NodeType.DECISION
         and n.status == NodeStatus.COMPLETED
         and not n._evicted],
        key=lambda x: x.importance, reverse=True
    )
    if decisions:
        _add(words["decided"], [d.label + _rationale(d) for d in decisions[:6]], " | ", prio=3)

    # ── 5b. Contested decisions — same topic, ambiguous whether one replaces
    # the other (see NodeStatus.CONTESTED). Surfaced explicitly
    # rather than silently guessing which one is "current" — unlike
    # SUPERSEDED, both sides stay visible here since destroying either
    # one would risk losing correct information on weak evidence.
    contested = [
        n for n in graph._nodes.values()
        if n.type == NodeType.DECISION
        and n.status == NodeStatus.CONTESTED
        and not n._evicted
    ]
    if contested:
        contested_ids = {n.id for n in contested}
        seen_pairs: set[frozenset] = set()
        lines = []
        for e in graph._edges:
            if (e.type == EdgeType.CONFLICTS_WITH
                    and e.source_id in contested_ids and e.target_id in contested_ids):
                pair = frozenset((e.source_id, e.target_id))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                a, b = graph._nodes[e.source_id], graph._nodes[e.target_id]
                lines.append(f"{a.label[:45]!r} vs {b.label[:45]!r}")
        if lines:
            _add("Conflicting (unresolved)", lines[:3], " | ", prio=1)

    # ── 6. Decision transitions — compact, no wasted tokens on wrong answer ─
    # Show as "Changed X → Y" not full old label — the old label is wrong,
    # showing it in full wastes tokens and risks LLM being confused about
    # which is current.
    recent_transitions = sorted(
        graph._transitions,
        key=lambda t: t.timestamp, reverse=True
    )[:3]
    if recent_transitions:
        lines = [t.to_context_line() for t in recent_transitions]
        _add("Changes", lines, " | ", prio=4)
    elif any(
        n.type == NodeType.DECISION
        and n.status == NodeStatus.SUPERSEDED
        and n.age_days() < 3
        and not n._evicted
        for n in graph._nodes.values()
    ):
        # No transition object but recent supersede — note count only, no label
        # (showing the old wrong label wastes tokens and risks LLM confusion)
        changed_count = sum(
            1 for n in graph._nodes.values()
            if n.type == NodeType.DECISION
            and n.status == NodeStatus.SUPERSEDED
            and n.age_days() < 3
            and not n._evicted
        )
        _add("Note", [f"{changed_count} decision(s) changed recently — see graph history"],
             " | ", prio=4)

    # ── 7. Invalidated decisions — always warn ─────────────────────────────
    invalidated = [
        n for n in graph._nodes.values()
        if n.type == NodeType.DECISION
        and n.status == NodeStatus.INVALIDATED
        and not n._evicted
    ]
    if invalidated:
        _add("Avoid", [f"[DO NOT USE] {n.label[:40]}" for n in invalidated[:2]], " | ", prio=1)

    # ── 7b. Constraints from windowed-out turns ───────────────────────────
    # Placed above Files deliberately: a file list is re-derivable from the
    # repository, and "keep the bundle under 500KB" is not re-derivable
    # from anything once the turn that said it has left the conversation.
    # See summary.py for why these sentences exist as a node at all.
    notes = sorted(
        [n for n in graph._nodes.values()
         if n.type == NodeType.SUMMARY and not n._evicted],
        key=lambda x: x.updated_at, reverse=True
    )
    if notes:
        # "Noted:" and not "Constraints:": the selector keeps hard figures
        # as well as rules ("21 percent of the suite fails here"), and a
        # header that overpromises is how a reader learns to distrust one.
        _add("Noted", [notes[0].label], " | ", prio=6)

    # ── 8. Files ──────────────────────────────────────────────────────────
    # Edited files first (higher importance), then the most recently
    # touched: on importance alone every file tied and the section listed
    # the files named EARLIEST in the session.
    files = sorted(
        [n for n in graph._nodes.values()
         if n.type == NodeType.FILE and not n._evicted],
        key=_recent_first, reverse=True
    )
    if files:
        _add(words["files"], [_short_path(f.label) for f in files[:10]], ", ", prio=7)

    # ── 9. Environment ────────────────────────────────────────────────────
    env_nodes = [
        n for n in graph._nodes.values()
        if n.type == NodeType.ENVIRONMENT and not n._evicted
    ]
    if env_nodes:
        _add("Env", [e.label for e in env_nodes[:4]], ", ", prio=8)

    # ── 10. Open errors ───────────────────────────────────────────────────
    #
    # Most recent first, like work in progress: every open error has the
    # same importance, so sorting on importance alone fell back to insertion
    # order and the resume listed the OLDEST three failures. On real
    # SWE-bench sessions that is the first traceback of the session, not
    # the one the agent was stuck on when it stopped.
    errors = sorted(
        [n for n in graph._nodes.values()
         if n.type == NodeType.ERROR
         and n.status == NodeStatus.FAILED
         and not n._evicted],
        key=_recent_first, reverse=True
    )
    if errors:
        _add(words["errors"], [e.label for e in errors[:3]], " | ", prio=1)

    return _pack(sections, token_budget)


# ── Packing ──────────────────────────────────────────────────────────────────


def _pack(sections: list, token_budget: int) -> str:
    """Fit the sections into `token_budget`, item by item.

    This used to build every section in full and then, while over budget,
    drop WHOLE sections from the bottom — so the first thing a tight budget
    lost was "Open issues", the section the rest of this module calls the
    most important thing to carry into a resume, and every file at once.
    Measured on the external benchmark at a 150-token budget, open errors
    survived in 41% of sessions.

    Items are admitted in rounds instead: the first item of every section,
    most important section first; then every second item; and so on. A
    section keeps its place in the display order whatever its priority.
    An item that does not fit is skipped and smaller ones after it can
    still go in. Costs are counted per item and the result is verified
    with one exact count at the end.
    """
    from tokenmizer.core.tokenizer import count_tokens

    if not sections:
        return ""
    budget = max(0, token_budget)
    chosen: list[list[str]] = [[] for _ in sections]
    used = 0
    order = sorted(range(len(sections)), key=lambda i: sections[i][3])
    longest = max(len(sec[1]) for sec in sections)
    for rnd in range(longest):
        for i in order:
            header, items, sep, _prio = sections[i]
            if rnd >= len(items):
                continue
            item = items[rnd]
            lead = header + ": " if not chosen[i] else sep
            cost = count_tokens(lead + item) + (1 if not chosen[i] else 0)
            if used + cost > budget:
                continue
            chosen[i].append(item)
            used += cost

    def render() -> str:
        return "\n".join(
            sections[i][0] + ": " + sections[i][2].join(chosen[i])
            for i in range(len(sections)) if chosen[i]
        )

    block = render()
    # Per-item counts can undercount where two items tokenise differently
    # side by side; drop the least important admitted item until the exact
    # count agrees.
    while block and count_tokens(block) > budget:
        worst = max((i for i in range(len(sections)) if chosen[i]),
                    key=lambda i: (sections[i][3], len(chosen[i])))
        chosen[worst].pop()
        block = render()
    return block


_HOME = re.compile(r"^(?:/home/[^/]+|/Users/[^/]+|/root|[A-Za-z]:\\Users\\[^\\]+)(?=[/\\])")


def _short_path(path: str) -> str:
    """`/home/dev/app/src/x.py` -> `~/app/src/x.py`. An agent's tool calls
    name files by absolute path, and the home prefix is the same few tokens
    on every file in the list while saying nothing about which file."""
    return _HOME.sub("~", path, count=1)


def _rationale(decision) -> str:
    """The decision's reason as " (...)" when it is short and adds something.

    The summary used to be appended cut at 50 characters, mid-word, and it
    is often the sentence the decision was extracted FROM — which restates
    the label and spends tokens on a fragment.
    """
    from tokenmizer.graph_memory.patterns import _clip

    summary = (decision.summary or "").strip()
    if not summary or "Superseded by" in summary:
        return ""
    reason = _clip(summary, 60)
    if len(reason) < 8:
        return ""
    label_words = set(re.findall(r"[a-z0-9]+", decision.label.lower()))
    reason_words = set(re.findall(r"[a-z0-9]+", reason.lower()))
    if reason_words and len(reason_words & label_words) / len(reason_words) >= 0.5:
        return ""
    return f" ({reason})"
