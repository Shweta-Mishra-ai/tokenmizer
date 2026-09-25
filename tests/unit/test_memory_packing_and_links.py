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


def test_a_file_is_linked_only_when_named_as_a_whole_word(tmp_path):
    # A substring test linked api.py to "rapid", app.py to "happy" and an
    # error in data.py to lib/a.py.
    g = GraphMemory("s", storage_dir=str(tmp_path))
    _feed(g, [
        {"role": "assistant", "content": "I edited src/api.py and src/app.py."},
        {"role": "assistant", "content": "Next step: plan the rapid rollout for the happy path."},
        {"role": "assistant", "content": "TypeError in data.py: bad value"},
        {"role": "assistant", "content": "Also touched lib/a.py and src/orders.py."},
        {"role": "assistant", "content": "Next step: add pagination to the orders endpoint."},
    ])
    implements = _edges(g, EdgeType.IMPLEMENTS)
    related = _edges(g, EdgeType.RELATED_TO)
    assert not any(f in ("src/api.py", "src/app.py") for _, f in implements), implements
    assert ("TypeError in data.py", "lib/a.py") not in related, related
    assert any(f == "src/orders.py" for _, f in implements), implements


# ── Long sessions: the processed-hash cap ────────────────────────────────────

def test_messages_older_than_the_hash_window_are_not_re_extracted(tmp_path, monkeypatch):
    # Past the cap the set is trimmed to the most recent messages. Every
    # older message used to look new again, so each later request re-ran
    # extraction over all of them: on a real 914-message agent session,
    # 414 messages per request.
    from tokenmizer.graph_memory import graph as graph_mod
    monkeypatch.setattr(graph_mod, "_MAX_PROCESSED_HASHES", 20)
    msgs = [{"role": "user" if i % 2 else "assistant", "content": f"note number {i}"}
            for i in range(40)]
    g = GraphMemory("s", storage_dir=str(tmp_path))
    for k in range(1, 31):
        g.extract_from_messages(msgs[:k])

    seen = []
    real = g._apply_extracted
    monkeypatch.setattr(g, "_apply_extracted",
                        lambda data, new: (seen.append(len(new)), real(data, new))[1])
    for k in range(31, 41):
        g.extract_from_messages(msgs[:k])
    assert seen == [1] * 10, seen


def test_the_trimmed_window_survives_a_restart(tmp_path, monkeypatch):
    from tokenmizer.graph_memory import graph as graph_mod
    monkeypatch.setattr(graph_mod, "_MAX_PROCESSED_HASHES", 20)
    msgs = [{"role": "assistant", "content": f"message {i}"} for i in range(30)]
    g = GraphMemory("s", storage_dir=str(tmp_path))
    for k in range(1, 31):
        g.extract_from_messages(msgs[:k])
    g._persist(force=True)

    reloaded = GraphMemory("s", storage_dir=str(tmp_path))
    assert graph_mod._HASHES_TRIMMED in reloaded._processed_hashes
    seen = []
    real = reloaded._apply_extracted
    monkeypatch.setattr(reloaded, "_apply_extracted",
                        lambda data, new: (seen.append(len(new)), real(data, new))[1])
    reloaded.extract_from_messages(msgs + [{"role": "assistant", "content": "OSError: [Errno 28] No space left on device"}])
    assert seen == [1], seen
    assert any(n.type == NodeType.ERROR for n in reloaded._nodes.values())
