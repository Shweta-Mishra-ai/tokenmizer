"""
Unit tests — graph_memory/visualization.py.

Previously zero tests for this entire module (to_vis_json,
to_obsidian_canvas, to_share_html) despite to_share_html being served
directly over HTTP as raw HTML (GET /api/graph/{session_id}/html).

Main focus: to_share_html() built its output with a single unescaped
string substitution reused across THREE different sink contexts (two
plain-HTML text spots, one JS string literal) for session_id — which is
client-supplied — plus json.dumps() output embedded in a <script> block
without guarding the "</script>" breakout, for node/decision text that
comes straight from conversation content. The endpoint is auth-gated,
but its own docstring says "share": the intended use is downloading the
file and handing it to someone else, so a payload here isn't self-XSS,
it runs in whoever opens the shared file.
"""
from __future__ import annotations

from tokenmizer.graph_memory.graph import GraphMemory, NodeStatus, NodeType
from tokenmizer.graph_memory.types import EdgeType
from tokenmizer.graph_memory.visualization import (
    _EDGE_COLOR,
    _TYPE_COLOR,
    _TYPE_SIZE,
    to_obsidian_canvas,
    to_share_html,
    to_vis_json,
)


class TestColorMaps:
    """The maps are the only place a node type gets a color, and
    to_vis_json's meta.by_type iterates _TYPE_COLOR — a type missing from
    it was silently dropped from the counts as well as drawn grey."""

    def test_every_node_type_has_a_color(self):
        assert set(_TYPE_COLOR) == {t.value for t in NodeType}

    def test_every_node_type_has_a_size(self):
        assert set(_TYPE_SIZE) == {t.value for t in NodeType}

    def test_edge_colors_match_edge_type(self):
        assert set(_EDGE_COLOR) == {e.value for e in EdgeType}

    def test_type_colors_are_distinct(self):
        assert len(set(_TYPE_COLOR.values())) == len(_TYPE_COLOR)


def _graph(tmp_path, session_id="t-viz"):
    g = GraphMemory(session_id=session_id, storage_dir=str(tmp_path))
    g.add_node(NodeType.GOAL, "Build auth service", NodeStatus.IN_PROGRESS)
    g.add_node(NodeType.DECISION, "Use PostgreSQL for storage",
              NodeStatus.COMPLETED, summary="concurrent writes")
    g.add_node(NodeType.FILE, "api/auth.py", NodeStatus.COMPLETED)
    return g


class TestToVisJson:

    def test_basic_shape(self, tmp_path):
        vis = to_vis_json(_graph(tmp_path))
        assert vis["session_id"] == "t-viz"
        assert len(vis["nodes"]) == 3
        assert vis["meta"]["node_count"] == 3
        types = {n["type"] for n in vis["nodes"]}
        assert types == {"goal", "decision", "file"}

    def test_evicted_nodes_are_excluded(self, tmp_path):
        g = _graph(tmp_path)
        nid = next(iter(g._nodes))
        g._nodes[nid]._evicted = True
        vis = to_vis_json(g)
        assert nid not in {n["id"] for n in vis["nodes"]}

    def test_nodes_carry_community_and_meta_lists_them(self, tmp_path):
        from tokenmizer.graph_memory.types import EdgeType

        g = _graph(tmp_path)
        ids = {n.type.value: nid for nid, n in g._nodes.items()}
        g.add_edge(ids["goal"], ids["decision"], EdgeType.IMPLEMENTS)
        vis = to_vis_json(g)

        assert all("community" in n for n in vis["nodes"])
        coms = vis["meta"]["communities"]
        assert coms[0]["count"] == 2, "goal and decision are linked, so they cluster"
        assert {"id", "name", "color", "count"} <= set(coms[0])
        assert coms[0]["name"] in {"Build auth service", "Use PostgreSQL for storage"}
        assert coms[-1]["name"] == "Unclustered"
        assert coms[-1]["count"] == 1

    def test_meta_carries_the_health_fields_the_page_needs(self, tmp_path):
        vis = to_vis_json(_graph(tmp_path))
        meta = vis["meta"]
        assert meta["processed_messages"] == 0
        assert meta["load_failed"] is False
        assert meta["persistence_broken"] is False
        assert meta["data_loss_detected"] is False


class TestToObsidianCanvas:

    def test_basic_shape(self, tmp_path):
        canvas = to_obsidian_canvas(_graph(tmp_path))
        # +1 for the legend node inserted at index 0
        assert len(canvas["nodes"]) == 4
        assert canvas["nodes"][0]["id"] == "legend"

    def test_empty_graph_does_not_crash(self, tmp_path):
        g = GraphMemory(session_id="t-empty", storage_dir=str(tmp_path))
        canvas = to_obsidian_canvas(g)
        assert canvas["nodes"][0]["id"] == "legend"
        assert canvas["edges"] == []


class TestToShareHtmlEscaping:

    def test_session_id_is_html_escaped_in_page_text(self, tmp_path):
        g = GraphMemory(session_id='"><script>alert(1)</script>',
                        storage_dir=str(tmp_path))
        html = to_share_html(g)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html

    def test_session_id_cannot_break_out_of_the_js_string(self, tmp_path):
        """The PNG-download filename used to be built by substituting the
        raw session_id straight into a JS double-quoted string literal —
        a session_id containing '"' broke out of it. It now reads
        DATA.session_id (a JSON-embedded value) at runtime instead, so
        the template must contain no literal, substituted session_id
        inside a JS string context at all."""
        g = GraphMemory(session_id='x";alert(1);//', storage_dir=str(tmp_path))
        html = to_share_html(g)
        assert 'x";alert(1);//' not in html
        assert "DATA.session_id" in html

    def test_conversation_content_cannot_close_the_script_tag(self, tmp_path):
        """A decision label containing the literal text "</script>"
        (e.g. someone discussing or pasting HTML/JS in the conversation
        that got extracted into a node) must not terminate the enclosing
        <script> block — json.dumps alone does not escape "</"."""
        g = GraphMemory(session_id="t-esc", storage_dir=str(tmp_path))
        g.add_node(NodeType.DECISION,
                  'Use </script><img src=x onerror=alert(1)> for parsing',
                  NodeStatus.COMPLETED)
        html = to_share_html(g)
        assert "<img src=x onerror=alert(1)>" not in html
        assert "</script><img" not in html
        # Escaping "<" alone is sufficient — the browser's HTML tokenizer
        # only ends a <script> block on a literal "<" starting "</script",
        # so a "\u003c" there can never be read as one regardless of what
        # follows. The content itself is preserved, just neutralized, not
        # silently dropped: the escaped form is present in the output.
        assert "\\u003c/script>" in html

    def test_output_still_contains_real_data(self, tmp_path):
        """The escaping fix must not have broken the actual export."""
        html = to_share_html(_graph(tmp_path))
        assert "Use PostgreSQL for storage" in html
        assert '"nodes"' in html
        assert "t-viz" in html


class TestShareHtmlPage:
    """What the page must carry for the community explorer and the
    empty state. The empty state is the important one: an empty graph
    used to render a blank canvas with "0 nodes" and no explanation."""

    PRESERVED = ["Decision history", '"transitions"', "Active only",
                 "exportPng", "DATA.session_id", '"nodes"']

    def test_populated_page_has_communities_and_detail_panel(self, tmp_path):
        html = to_share_html(_graph(tmp_path))
        for literal in self.PRESERVED:
            assert literal in html, literal
        assert '"communities"' in html
        assert "Communities" in html
        assert "Select all" in html
        assert 'id="detail"' in html
        assert "<script src=" not in html and "https://cdn" not in html

    def test_empty_page_explains_itself(self, tmp_path):
        g = GraphMemory(session_id="t-empty-page", storage_dir=str(tmp_path))
        html = to_share_html(g)
        assert "No nodes yet" in html
        assert '"processed_messages"' in html
        assert '"load_failed"' in html
        for literal in self.PRESERVED:
            assert literal in html, literal

    def test_no_entity_emoji_in_the_template(self, tmp_path):
        html = to_share_html(_graph(tmp_path))
        assert "&#129504;" not in html
