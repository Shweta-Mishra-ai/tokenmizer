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
from tokenmizer.graph_memory.visualization import (
    to_obsidian_canvas,
    to_share_html,
    to_vis_json,
)


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
