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
    _TYPE_COLOR_LIGHT,
    _TYPE_ORDER,
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

    # The eight categorical slots, in the order the palette validator was
    # run on. Pinned so a hex cannot be changed by eye: re-run
    #   node scripts/validate_palette.js "<the eight>" --mode dark --surface "#12141c"
    # (and --mode light --surface "#f7f8fc" for the light set) and paste
    # the new values here with the result.
    CATEGORICAL = ["file", "endpoint", "task", "dependency",
                   "goal", "schema", "decision", "error"]
    VALIDATED_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
                      "#d55181", "#008300", "#9085e9", "#e66767"]
    VALIDATED_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                       "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

    def test_the_categorical_slots_are_the_validated_ones(self):
        """The old palette put goal (#e879f9) and decision (#a78bfa) 0.4
        apart under protanopia and 10.9 apart for a full-colour reader, so
        the two types the product is built around were indistinguishable
        for a large share of people. These eight passed every check on
        both surfaces; do not substitute a hex without re-running the
        validator."""
        assert [_TYPE_COLOR[t] for t in self.CATEGORICAL] == self.VALIDATED_DARK
        assert [_TYPE_COLOR_LIGHT[t] for t in self.CATEGORICAL] == self.VALIDATED_LIGHT

    def test_the_categorical_slots_are_distinct(self):
        assert len(set(self.VALIDATED_DARK)) == 8
        assert len(set(self.VALIDATED_LIGHT)) == 8

    def test_types_beyond_the_eight_slots_share_one_neutral(self):
        """A ninth series is folded into "other", never given a generated
        hue — eight is the most a categorical palette can carry and still
        be told apart. Position and the node label carry these types."""
        rest = [t for t in _TYPE_COLOR if t not in self.CATEGORICAL]
        assert rest, "the map should still cover every NodeType"
        assert len(set(_TYPE_COLOR[t] for t in rest)) == 1
        assert len(set(_TYPE_COLOR_LIGHT[t] for t in rest)) == 1

    def test_both_themes_cover_every_type(self):
        assert set(_TYPE_COLOR_LIGHT) == set(_TYPE_COLOR)

    def test_every_type_has_a_place_in_the_layout_order(self):
        """The radial view reads type off WHICH ARC a node sits on, so a
        type missing from the order has no arc and no position — which is
        the encoding that makes the palette legal in the first place."""
        assert set(_TYPE_ORDER) == {t.value for t in NodeType}


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


class TestDerivedAnalytics:
    """The panel's numbers are computed here, not in the browser, so they
    are testable and identical for every consumer of the export."""

    def _graph_with_history(self, tmp_path):
        g = GraphMemory(session_id="t-analytics", storage_dir=str(tmp_path))
        g.extract_from_messages([
            {"role": "user", "content": "We are building the checkout service"},
            {"role": "assistant", "content":
             "Decided: PostgreSQL for order storage. Created internal/store/postgres.go."},
            {"role": "assistant", "content":
             "Switched from moment.js to date-fns. The build fails with exit code 1."},
        ], incremental=False)
        return g

    def test_counts_describe_the_session(self, tmp_path):
        meta = to_vis_json(self._graph_with_history(tmp_path))["meta"]
        counts = meta["counts"]
        assert counts["nodes"] == meta["node_count"]
        assert counts["relations"] == meta["edge_count"]
        assert counts["decisions"] >= 1
        assert counts["changes"] == meta["transition_count"] >= 1
        assert counts["open_issues"] >= 1

    def test_hotspots_rank_by_what_depends_on_a_node(self, tmp_path):
        g = GraphMemory(session_id="t-hot", storage_dir=str(tmp_path))
        goal = g.add_node(NodeType.GOAL, "Ship the checkout service",
                          NodeStatus.IN_PROGRESS)
        for i in range(3):
            task = g.add_node(NodeType.TASK, f"Completed step number {i}",
                              NodeStatus.COMPLETED)
            g.add_edge(task, goal, EdgeType.PART_OF)
        hotspots = to_vis_json(g)["meta"]["hotspots"]
        assert hotspots[0]["label"] == "Ship the checkout service"
        assert hotspots[0]["depends_on_it"] == 3
        assert all("id" not in h for h in hotspots), "internal ids stay internal"

    def test_flows_count_relations_between_types(self, tmp_path):
        meta = to_vis_json(self._graph_with_history(tmp_path))["meta"]
        pairs = {(f["source"], f["target"]): f["count"] for f in meta["flows"]}
        assert pairs, "a session with edges must report flows"
        assert sum(pairs.values()) <= meta["edge_count"]
        counts = [f["count"] for f in meta["flows"]]
        assert counts == sorted(counts, reverse=True), "flows are ranked"

    def test_history_gaps_are_reported(self, tmp_path):
        """A superseded decision with no transition means `why` has a hole
        in it. It was only visible by calling /reasoning and reading a
        list of anomalies."""
        g = GraphMemory(session_id="t-gap", storage_dir=str(tmp_path))
        nid = g.add_node(NodeType.DECISION, "Use the old approach",
                         NodeStatus.COMPLETED)
        g._nodes[nid].status = NodeStatus.SUPERSEDED
        assert to_vis_json(g)["meta"]["counts"]["unexplained_supersessions"] == 1

    def test_a_clean_session_reports_no_gaps(self, tmp_path):
        meta = to_vis_json(self._graph_with_history(tmp_path))["meta"]
        assert meta["counts"]["dangling_history"] == 0
        assert meta["counts"]["unexplained_supersessions"] == 0

    def test_empty_graph_analytics_do_not_crash(self, tmp_path):
        meta = to_vis_json(GraphMemory("t-empty-an", storage_dir=str(tmp_path)))["meta"]
        assert meta["counts"]["nodes"] == 0
        assert meta["hotspots"] == [] and meta["flows"] == []

    def test_nodes_are_exported_in_layout_order(self, tmp_path):
        """The radial view reads type off position, so the arcs have to be
        contiguous — which means the export, not the browser, decides the
        order."""
        g = self._graph_with_history(tmp_path)
        types = [n["type"] for n in to_vis_json(g)["nodes"]]
        first_seen = []
        for t in types:
            if t not in first_seen:
                first_seen.append(t)
        assert types == sorted(types, key=lambda t: first_seen.index(t)), \
            "nodes of one type must be contiguous"
        rank = {t: i for i, t in enumerate(_TYPE_ORDER)}
        assert first_seen == sorted(first_seen, key=lambda t: rank[t])


class TestRadialViewContract:
    """What the page needs in order to render the radial layout at all."""

    def test_the_page_defaults_to_radial(self, tmp_path):
        html = to_share_html(_graph(tmp_path))
        assert 'id="viewRadial"' in html
        assert 'let alpha=1, mode="radial"' in html

    def test_the_page_carries_both_palettes(self, tmp_path):
        html = to_share_html(_graph(tmp_path))
        assert "COLOR_DARK" in html and "COLOR_LIGHT" in html
        assert "#9085e9" in html and "#4a3aa7" in html

    def test_meta_carries_the_layout_order(self, tmp_path):
        meta = to_vis_json(_graph(tmp_path))["meta"]
        assert meta["type_order"]
        assert set(meta["type_order"]) <= set(_TYPE_ORDER)


class TestForceViewContract:
    """The force view was rebuilt three times before it was readable.
    What each round fixed is pinned here, because the failures are
    invisible to a test that only asks whether the page renders.
    """

    def test_the_layout_is_settled_before_anything_is_drawn(self, tmp_path):
        """The old code reframed the view on a 260ms timer while the
        simulation still had seconds of travel left, so the picture the
        reader got was framed for a layout that no longer existed —
        nodes ran off the bottom and under the side panel."""
        html = to_share_html(_graph(tmp_path))
        assert "function settle()" in html
        assert "setTimeout(fit" not in html

    def test_unconnected_nodes_are_placed_not_simulated(self, tmp_path):
        """A node with no edge feels repulsion from everything and has
        no spring pulling back, so the cluster fires it at the canvas
        wall and it stays there. Seven of them made a border of exiles."""
        html = to_share_html(_graph(tmp_path))
        assert "function placeIsolates()" in html
        assert "const sim=nodes.filter(a=>nbrs[a.id].length)" in html
        assert "not linked to anything yet" in html

    def test_the_forces_are_fruchterman_reingold(self, tmp_path):
        html = to_share_html(_graph(tmp_path))
        assert "const f=K2/d" in html, "repulsion must be K²/d, uncapped"
        assert "const f=(d*d)/K" in html, "attraction must be d²/K"

    def test_cohesion_is_written_in_the_same_units_as_the_spring(
            self, tmp_path):
        """As a linear force it contributed single digits against a
        spring in the hundreds, so communities never gathered and the
        hull drawn round one spanned half the canvas."""
        html = to_share_html(_graph(tmp_path))
        assert "const f=(d*d)/K*w" in html

    def test_fit_accounts_for_the_label_text(self, tmp_path):
        """Fitting to the dots alone pushed every right-hand label under
        the side panel."""
        html = to_share_html(_graph(tmp_path))
        assert 'if(mode!=="radial"&&!PREVIEW){' in html
