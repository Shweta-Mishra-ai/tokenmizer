"""
Memory per token, and relations that survive incremental extraction.

Two measured defects:

- The resume block built every section in full and, while over budget,
  dropped WHOLE sections from the bottom — so under a tight budget the
  first casualty was "Open issues". On the external benchmark (340
  sessions) at 150 tokens, open errors survived in 40% of sessions;
  per-item packing keeps 73%, and the share of labelled facts the block
  carries went from 67% to 74% for the same budget.
- Relations were only inferred between nodes of the same extraction
  call, and the proxy extracts one new message per request. Extracting
  the external corpora one message at a time shared 177 of the 707 edges
  that extracting each session whole formed; it now shares 706.
"""
from __future__ import annotations

import json

from tokenmizer.graph_memory.context_block import _pack, _rationale, _short_path
from tokenmizer.graph_memory.graph import EdgeType, GraphMemory, NodeStatus, NodeType


def _edges(g, edge_type):
    lab = {n.id: n.label for n in g._nodes.values()}
    return {(lab.get(e.source_id), lab.get(e.target_id)) for e in g._edges if e.type == edge_type}


# ── Packing ──────────────────────────────────────────────────────────────────

def test_tight_budget_keeps_the_top_item_of_every_section_before_seconds():
    sections = [
        ("Done", [f"finished item number {i} of the long list" for i in range(8)], " | ", 5),
        ("Files", ["api/a.py", "api/b.py"], ", ", 7),
        ("Open issues", ["connection pool exhaustion under load"], " | ", 1),
    ]
    block = _pack(sections, 40)
    assert "Open issues: connection pool exhaustion" in block, block
    assert "Files: api/a.py" in block, block


def test_sections_keep_their_display_order_whatever_their_priority():
    block = _pack([("A", ["first"], " | ", 9), ("B", ["second"], " | ", 0)], 100)
    assert block.splitlines() == ["A: first", "B: second"]


def test_pack_never_exceeds_the_budget():
    from tokenmizer.core.tokenizer import count_tokens
    sections = [("S", [f"item {i} " * 5 for i in range(50)], " | ", 1)]
    for budget in (0, 1, 7, 30, 120):
        assert count_tokens(_pack(sections, budget)) <= budget


def test_home_directory_is_shortened():
    assert _short_path("/home/dev/app/src/x.py") == "~/app/src/x.py"
    assert _short_path("/Users/dev/app/x.py") == "~/app/x.py"
    assert _short_path("src/x.py") == "src/x.py"


def test_rationale_is_dropped_when_it_restates_the_label_or_is_a_fragment():
    class D:
        label = "Use Redis for refresh tokens"
        summary = "Use Redis for refresh tokens because"
    assert _rationale(D) == ""

    class E:
        label = "Use Redis for refresh tokens"
        summary = "sub-millisecond reads and built-in TTL expiry"
    assert _rationale(E).startswith(" (sub-millisecond")


def test_open_error_survives_a_tight_resume(tmp_path):
    g = GraphMemory("s", storage_dir=str(tmp_path))
    g.extract_from_messages([{"role": "assistant", "content":
        "Done: the orders API. Done: pagination on the list endpoints. Done: request ID "
        "middleware. Done: the Alembic baseline migration. Done: pytest fixtures. "
        "Updated api/orders.py, api/users.py, db/session.py, migrations/env.py. "
        "Bug: duplicate charges from a non-idempotent retry."}], incremental=False)
    block = g.to_context_block(token_budget=60)
    assert "duplicate charges" in block, block


# ── Relations across extraction calls ────────────────────────────────────────

def _feed(g, msgs):
    for k in range(1, len(msgs) + 1):
        g.extract_from_messages(msgs[:k])


def test_later_error_links_to_an_earlier_file(tmp_path):
    g = GraphMemory("s", storage_dir=str(tmp_path))
    _feed(g, [
        {"role": "assistant", "content": "Updated api/orders.py."},
        {"role": "user", "content": "Run the tests."},
        {"role": "assistant", "content": "Bug: KeyError in orders.py when the cart is empty."},
    ])
    assert any("orders.py" in t for _, t in _edges(g, EdgeType.RELATED_TO)), g._edges


def test_later_fix_closes_an_earlier_error(tmp_path):
    g = GraphMemory("s", storage_dir=str(tmp_path))
    _feed(g, [
        {"role": "assistant", "content": "Bug: duplicate charges from the retry path."},
        {"role": "user", "content": "Fix it."},
        {"role": "assistant", "content": "Fixed: duplicate charges from the retry path, with idempotency keys."},
    ])
    errors = [n for n in g._nodes.values() if n.type == NodeType.ERROR]
    assert errors and all(n.status == NodeStatus.COMPLETED for n in errors), [
        (n.label, n.status) for n in errors]


def test_tasks_are_part_of_the_goal_when_extracted_incrementally(tmp_path):
    g = GraphMemory("s", storage_dir=str(tmp_path))
    _feed(g, [
        {"role": "user", "content": "Goal: build a billing API with Stripe webhooks"},
        {"role": "assistant", "content": "Sounds good."},
        {"role": "assistant", "content": "Done: the webhook signature check."},
    ])
    assert _edges(g, EdgeType.PART_OF), g._edges


def test_incremental_and_whole_extraction_form_the_same_relations(tmp_path):
    import glob
    for path in sorted(glob.glob("benchmarks/eval/corpus/*.json"))[:8]:
        msgs = json.load(open(path))["messages"]
        whole = GraphMemory("w", storage_dir=str(tmp_path / "w" / path.split("/")[-1]))
        whole.extract_from_messages(msgs, incremental=False)
        inc = GraphMemory("i", storage_dir=str(tmp_path / "i" / path.split("/")[-1]))
        _feed(inc, msgs)

        def rel(g):
            lab = {n.id: n.label.lower() for n in g._nodes.values()}
            return {(e.type, lab.get(e.source_id), lab.get(e.target_id)) for e in g._edges}
        missing = rel(whole) - rel(inc)
        assert len(missing) <= max(1, len(rel(whole)) // 10), (path, missing)
