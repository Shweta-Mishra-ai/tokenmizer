"""
Resume context block — the tiered, token-budgeted summary injected into
the LLM at the start of a resumed session.

Extracted from graph.py to keep that file focused on core memory logic
(node/edge CRUD, extraction application, query, persistence). Follows
the same split pattern already established for visualization.py,
pruning.py, and persistence.py: a module-level function taking
`graph: GraphMemory` as its argument, with GraphMemory keeping a
one-line delegating method so existing callers are unaffected.

Pure code motion — no logic changes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from tokenmizer.graph_memory.types import EdgeType, NodeStatus, NodeType

if TYPE_CHECKING:
    from tokenmizer.graph_memory.graph import GraphMemory


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
    sections: list[str] = []

    # ── 1. Goal ──────────────────────────────────────────────────────────
    goals = sorted(
        [n for n in graph._nodes.values()
         if n.type == NodeType.GOAL and not n._evicted],
        key=lambda x: x.importance, reverse=True
    )
    if goals:
        sections.append("Goal: " + " | ".join(g.label for g in goals[:2]))

    # ── 2. In-progress tasks ──────────────────────────────────────────────
    open_tasks = sorted(
        [n for n in graph._nodes.values()
         if n.type == NodeType.TASK
         and n.status == NodeStatus.IN_PROGRESS
         and not n._evicted],
        key=lambda x: x.importance, reverse=True
    )
    # ── 3. Pending tasks (next steps) ─────────────────────────────────────
    pending_tasks = sorted(
        [n for n in graph._nodes.values()
         if n.type == NodeType.TASK
         and n.status == NodeStatus.PENDING
         and not n._evicted],
        key=lambda x: x.importance, reverse=True
    )
    current_work = open_tasks[:4] + pending_tasks[:2]
    if current_work:
        sections.append("Working on: " + " | ".join(t.label for t in current_work))

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
        sections.append("Done: " + " | ".join(t.label for t in done_deduped))

    # ── 5. Active decisions — top 6 by importance ─────────────────────────
    decisions = sorted(
        [n for n in graph._nodes.values()
         if n.type == NodeType.DECISION
         and n.status == NodeStatus.COMPLETED
         and not n._evicted],
        key=lambda x: x.importance, reverse=True
    )
    if decisions:
        parts = []
        for d in decisions[:6]:
            entry = d.label
            # Include brief rationale if not redundant with label
            if d.summary and "Superseded by" not in d.summary:
                entry += f" ({d.summary[:50]})"
            parts.append(entry)
        sections.append("Decided: " + " | ".join(parts))

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
            sections.append("Conflicting (unresolved): " + " | ".join(lines[:3]))

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
        sections.append("Changes: " + " | ".join(lines))
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
        sections.append(f"Note: {changed_count} decision(s) changed recently — see graph history")

    # ── 7. Invalidated decisions — always warn ─────────────────────────────
    invalidated = [
        n for n in graph._nodes.values()
        if n.type == NodeType.DECISION
        and n.status == NodeStatus.INVALIDATED
        and not n._evicted
    ]
    if invalidated:
        sections.append(
            "Avoid: " + " | ".join(f"[DO NOT USE] {n.label[:40]}" for n in invalidated[:2])
        )

    # ── 8. Files ──────────────────────────────────────────────────────────
    files = sorted(
        [n for n in graph._nodes.values()
         if n.type == NodeType.FILE and not n._evicted],
        key=lambda x: x.importance, reverse=True
    )
    if files:
        sections.append("Files: " + ", ".join(f.label for f in files[:10]))

    # ── 9. Environment ────────────────────────────────────────────────────
    env_nodes = [
        n for n in graph._nodes.values()
        if n.type == NodeType.ENVIRONMENT and not n._evicted
    ]
    if env_nodes:
        sections.append("Env: " + ", ".join(e.label for e in env_nodes[:4]))

    # ── 10. Open errors ───────────────────────────────────────────────────
    errors = sorted(
        [n for n in graph._nodes.values()
         if n.type == NodeType.ERROR
         and n.status == NodeStatus.FAILED
         and not n._evicted],
        key=lambda x: x.importance, reverse=True
    )
    if errors:
        sections.append("Open issues: " + " | ".join(e.label for e in errors[:3]))

    block = "\n".join(sections)

    # Trim to budget — count once, char-estimate for loop, exact verify at end
    from tokenmizer.core.tokenizer import count_tokens
    total_tokens = count_tokens(block)
    if total_tokens > token_budget and sections:

        chars_per_token = len(block) / max(total_tokens, 1)
        target_chars = int(token_budget * chars_per_token * 0.92)  # 8% safety buffer
        while len("\n".join(sections)) > target_chars and sections:
            sections.pop()
        block = "\n".join(sections)
        # One final accurate tiktoken verify — trim one more section if still over
        if sections and count_tokens(block) > token_budget:
            sections.pop()
            block = "\n".join(sections)

    return block
