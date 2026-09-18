"""
Graph visualization exports — D3, Obsidian Canvas.

Extracted from graph.py to keep that file focused on core memory logic.

Re-exported from graph.py for backward compatibility:
  from tokenmizer.graph_memory.graph import to_vis_json  (unchanged)
  graph.to_vis_json()  (unchanged — methods still on GraphMemory via import)
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenmizer.graph_memory.graph import GraphMemory


# ── Colour system ────────────────────────────────────────────────────────────
#
# VALIDATED, NOT CHOSEN BY EYE. The previous palette failed on the two node
# types that matter most: goal (#e879f9) and decision (#a78bfa) sat 0.4
# apart under protanopia and 10.9 apart for a full-colour reader, so a
# large share of people could not tell the product's headline type from
# its anchor. Re-run before changing any hex:
#
#   node scripts/validate_palette.js \
#     "#3987e5,#d95926,#199e70,#c98500,#d55181,#008300,#9085e9,#e66767" \
#     --mode dark --surface "#12141c"
#
# Results on record (OKLab ΔE ×100):
#   dark, adjacent pairs  — ALL PASS, worst CVD 8.4, worst normal 19.3
#   light, adjacent pairs — ALL PASS, worst CVD 9.1, worst normal 19.6
#   dark, ALL pairs       — worst CVD 1.6, worst normal 7.1  (FAILS)
#
# A node-link graph is an all-pairs form: any two types can end up side by
# side, and no ordering of eight hues clears the all-pairs floor. So colour
# is NOT the primary channel here, by design — this is the composite
# encoding the method allows:
#
#   1. POSITION. The default radial view gives every type its own labelled
#      arc, so type is read off where a node sits before any colour is.
#   2. A TEXT LABEL on every node, always, at every zoom.
#   3. A LEGEND naming each type with its swatch and count.
#
# Do not remove the arc labels or the per-node labels "to reduce clutter":
# they are what makes the palette legal.
#
# One entry per NodeType, enforced by tests/unit/test_visualization.py:
# to_vis_json's meta.by_type iterates this map, so a type missing here is
# not only drawn grey but dropped from the counts as well.
#
# Slot order below is the validated order. Types beyond the eight slots
# share a neutral: the method folds a ninth series into "other" rather
# than generating a hue, and these are the types a session rarely carries.
_NEUTRAL = "#8a8f9e"

_TYPE_COLOR = {
    "file":        "#3987e5",   # slot 1 blue
    "endpoint":    "#d95926",   # slot 2 orange
    "task":        "#199e70",   # slot 3 aqua
    "dependency":  "#c98500",   # slot 4 yellow
    "goal":        "#d55181",   # slot 5 magenta
    "schema":      "#008300",   # slot 6 green
    "decision":    "#9085e9",   # slot 7 violet
    # Slot 8 red, and semantically right: an ERROR node carries a status,
    # not an arbitrary series identity.
    "error":       "#e66767",
    "environment": _NEUTRAL,
    "concept":     _NEUTRAL,
    "api":         _NEUTRAL,
    "project":     _NEUTRAL,
    "agent":       _NEUTRAL,
    "test":        _NEUTRAL,
}

# The same eight hues stepped for a light surface, per the method: dark is
# a selected set, never an automatic flip. Three of them sit under 3:1 on
# the light surface, which the relief rule covers — every node carries a
# visible label and the legend names every type.
_TYPE_COLOR_LIGHT = {
    "file":        "#2a78d6",
    "endpoint":    "#eb6834",
    "task":        "#1baf7a",
    "dependency":  "#eda100",
    "goal":        "#e87ba4",
    "schema":      "#008300",
    "decision":    "#4a3aa7",
    "error":       "#e34948",
    "environment": "#6b7180",
    "concept":     "#6b7180",
    "api":         "#6b7180",
    "project":     "#6b7180",
    "agent":       "#6b7180",
    "test":        "#6b7180",
}

# Drawn radius in the radial view, before the importance term. A goal is
# the anchor of a session and a dependency is a name in a manifest.
_TYPE_SIZE = {
    "goal": 22, "decision": 18, "project": 16, "task": 14,
    "error": 14, "endpoint": 12, "schema": 12, "api": 12,
    "concept": 11, "agent": 11, "test": 10,
    "file": 10, "dependency": 9, "environment": 9,
}

# The order types are laid out in, around the circle and down the lanes:
# the order a session is read in, not alphabetical. Types absent from a
# session are skipped, so the arcs stay adjacent.
_TYPE_ORDER = (
    "goal", "decision", "task", "error", "endpoint",
    "schema", "file", "dependency", "environment",
    "test", "concept", "api", "project", "agent",
)

_STATUS_OPACITY = {
    "completed": 1.0, "in_progress": 0.9, "pending": 0.7,
    "failed": 0.6, "superseded": 0.35, "archived": 0.25,
    "modified": 0.5, "invalidated": 0.2,
    # Both sides of an unresolved conflict are live information (see
    # NodeStatus.CONTESTED), so they render at full weight.
    "contested": 1.0,
}

# One entry per EdgeType (same test). The previous map carried
# "references"/"derived_from", which no edge has ever had, and lacked
# fixes/blocks/conflicts_with, which fell through to the fallback grey.
_EDGE_COLOR = {
    "related_to":     "#8b8fa8",
    "implements":     "#60a5fa",
    "part_of":        "#a78bfa",
    "depends_on":     "#fbbf24",
    "supersedes":     "#f87171",
    "fixes":          "#4ade80",
    "blocks":         "#f87171",
    "conflicts_with": "#fb923c",
}

_CLUSTER_CENTERS: dict[str, tuple[float, float]] = {
    "goal":        (1500, 200),
    "decision":    (600,  700),
    "task":        (1500, 900),
    "file":        (2400, 700),
    "error":       (2400, 1400),
    "endpoint":    (600,  1400),
    "schema":      (1500, 1600),
    "dependency":  (300,  1200),
    "environment": (2700, 1200),
}

_TYPE_COLOR_OBS = {
    "goal": "6", "decision": "3", "task": "1",
    "file": "5", "error": "1", "endpoint": "4",
    "schema": "2", "dependency": "3", "environment": "4",
    "concept": "3", "api": "4", "project": "6", "agent": "1", "test": "2",
}

_EDGE_LABEL = {
    "related_to": "related", "implements": "implements",
    "part_of": "part of", "depends_on": "depends on",
    "supersedes": "supersedes", "fixes": "fixes",
    "blocks": "blocks", "conflicts_with": "conflicts with",
}


# Community colors for the graph page. Distinct from _TYPE_COLOR on purpose:
# the page fills a node by community and rings it by type, so the two
# palettes must not be confusable. Index mod len().
_COMMUNITY_PALETTE = (
    "#7c6af7", "#5ee7c8", "#f472b6", "#fbbf24", "#60a5fa", "#4ade80",
    "#fb923c", "#c084fc", "#22d3ee", "#a3e635", "#f87171", "#e879f9",
)
_UNCLUSTERED_COLOR = "#8b8fa8"
_UNCLUSTERED_NAME = "Unclustered"


def _communities(vis_nodes: list[dict], vis_edges: list[dict]) -> list[dict]:
    """Assign each node a community index (mutates the node dicts) and
    return the community list for meta. A community is named after its
    most important, best-connected member: the label a person would
    use to refer to that part of the graph."""
    from tokenmizer.graph_memory.communities import detect_communities

    assignment = detect_communities(
        [n["id"] for n in vis_nodes],
        [(e["source"], e["target"], e["weight"]) for e in vis_edges],
    )
    degree: dict[str, int] = {}
    for e in vis_edges:
        degree[e["source"]] = degree.get(e["source"], 0) + 1
        degree[e["target"]] = degree.get(e["target"], 0) + 1

    members: dict[int, list[dict]] = {}
    for n in vis_nodes:
        n["community"] = assignment[n["id"]]
        members.setdefault(n["community"], []).append(n)

    communities = []
    for index in sorted(members):
        group = members[index]
        singleton_group = all(degree.get(n["id"], 0) == 0 for n in group)
        if singleton_group:
            name, color = _UNCLUSTERED_NAME, _UNCLUSTERED_COLOR
        else:
            lead = min(group, key=lambda n: (-n["importance"], -degree.get(n["id"], 0), n["id"]))
            name = lead["full_label"][:40]
            color = _COMMUNITY_PALETTE[index % len(_COMMUNITY_PALETTE)]
        communities.append({"id": index, "name": name, "color": color, "count": len(group)})
    return communities


def _analytics(vis_nodes: list[dict], vis_edges: list[dict],
               transitions: list[dict]) -> dict:
    """The numbers the page's panel reports, derived here rather than in
    the browser so they are testable and identical for every consumer.

    Nothing here is a restatement of the node list. Each one answers a
    question a reader of a session graph actually has:

      hotspots   what does the rest of the session hang off? Ranked by
                 how many other nodes point AT a node, because that is
                 what makes it expensive to be wrong about.
      flows      which kinds of thing connect to which, counted. The
                 shape of the session in one table.
      dangling   transitions whose endpoints are no longer in the graph
                 (pruning can outrun history), and decisions marked
                 superseded with no transition explaining it. Both mean
                 `why` has a hole in it, and both were only visible by
                 calling the reasoning endpoint and reading a list.
    """
    by_id = {n["id"]: n for n in vis_nodes}
    in_degree: dict[str, int] = {n["id"]: 0 for n in vis_nodes}
    out_degree: dict[str, int] = {n["id"]: 0 for n in vis_nodes}
    flows: dict[tuple[str, str], int] = {}

    for e in vis_edges:
        source, target = by_id.get(e["source"]), by_id.get(e["target"])
        if source is None or target is None:
            continue
        out_degree[e["source"]] += 1
        in_degree[e["target"]] += 1
        key = (source["type"], target["type"])
        flows[key] = flows.get(key, 0) + 1

    # Ranked by what depends on a node, then by its own weight, then by
    # label so the order is stable for identical inputs.
    hotspots = sorted(
        (n for n in vis_nodes if in_degree[n["id"]]),
        key=lambda n: (-in_degree[n["id"]], -n["importance"], n["full_label"]),
    )[:8]

    node_ids = set(by_id)
    dangling = sum(
        1 for t in transitions
        if t["from_id"] not in node_ids or t["to_id"] not in node_ids
    )
    superseded_with_history = {t["from_id"] for t in transitions}
    unexplained = sum(
        1 for n in vis_nodes
        if n["type"] == "decision" and n["status"] in ("superseded", "modified")
        and n["id"] not in superseded_with_history
    )

    return {
        "counts": {
            "nodes": len(vis_nodes),
            "relations": len(vis_edges),
            "decisions": sum(1 for n in vis_nodes if n["type"] == "decision"),
            "changes": len(transitions),
            "open_issues": sum(1 for n in vis_nodes
                               if n["type"] == "error" and n["status"] == "failed"),
            "resolved_issues": sum(1 for n in vis_nodes
                                   if n["type"] == "error" and n["status"] == "completed"),
            "unlinked": sum(1 for n in vis_nodes
                            if not in_degree[n["id"]] and not out_degree[n["id"]]),
            "dangling_history": dangling,
            "unexplained_supersessions": unexplained,
        },
        "hotspots": [
            {"label": n["full_label"], "type": n["type"],
             "depends_on_it": in_degree[n["id"]], "importance": n["importance"]}
            for n in hotspots
        ],
        "flows": [
            {"source": source, "target": target, "count": count}
            for (source, target), count in sorted(
                flows.items(), key=lambda kv: (-kv[1], kv[0]))
        ][:8],
    }


def to_vis_json(graph: "GraphMemory") -> dict:
    """
    Export graph as D3-compatible JSON: {nodes, edges, transitions, meta}.
    Node colors and sizes encode type + status; `community` groups nodes
    by detected cluster (see communities.py), listed in meta.communities.
    Only exports non-evicted nodes. meta also carries the health fields
    the graph page needs to explain an empty graph honestly.
    """
    vis_nodes = []
    active_ids: set[str] = set()

    for nid, n in graph._nodes.items():
        if n._evicted:
            continue
        opacity = _STATUS_OPACITY.get(n.status.value, 0.8)
        vis_nodes.append({
            "id":         nid,
            "label":      n.label[:60] + ("\u2026" if len(n.label) > 60 else ""),
            "full_label": n.label,
            "type":       n.type.value,
            "status":     n.status.value,
            "importance": round(n.importance, 2),
            "confidence": round(n.confidence, 2),
            "summary":    n.summary or "",
            "age_days":   round(n.age_days(), 1),
            # Temporal fields: the page's timeline mode orders nodes by
            # when the fact became true, and the detail panel says when a
            # superseded one stopped being true. Exported as epoch
            # seconds, the same unit every other timestamp in the API uses.
            "created_at": round(n.created_at, 3),
            "updated_at": round(n.updated_at, 3),
            "valid_from": round(n.valid_from, 3),
            "valid_until": round(n.valid_until, 3) or None,
            "color":      _TYPE_COLOR.get(n.type.value, "#8b8fa8"),
            "size":       _TYPE_SIZE.get(n.type.value, 10),
            "opacity":    opacity,
        })
        active_ids.add(nid)

    vis_edges = []
    for e in graph._edges:
        if e.source_id not in active_ids or e.target_id not in active_ids:
            continue
        vis_edges.append({
            "source": e.source_id,
            "target": e.target_id,
            "type":   e.type.value,
            "weight": round(e.weight, 2),
            "color":  _EDGE_COLOR.get(e.type.value, "#4a4d5e"),
        })

    # Layout order for the radial view: types in reading order, and
    # within a type the heaviest first so an arc opens on what matters.
    order = {t: i for i, t in enumerate(_TYPE_ORDER)}
    vis_nodes.sort(key=lambda n: (order.get(n["type"], len(order)),
                                  -n["importance"], n["full_label"]))

    communities = _communities(vis_nodes, vis_edges)

    vis_transitions = [
        {
            "id":               t.id,
            "from_id":          t.from_decision_id,
            "to_id":            t.to_decision_id,
            "from_label":       t.from_label,
            "to_label":         t.to_label,
            "trigger":          t.trigger,
            "reason":           t.reason,
            "evidence":         t.evidence,
            "confidence_delta": t.confidence_delta,
            "timestamp":        t.timestamp,
        }
        for t in graph._transitions
    ]

    analytics = _analytics(vis_nodes, vis_edges, vis_transitions)

    return {
        "session_id":  graph.session_id,
        "nodes":       vis_nodes,
        "edges":       vis_edges,
        "transitions": vis_transitions,
        "meta": {
            **analytics,
            "node_count":       len(vis_nodes),
            "edge_count":       len(vis_edges),
            "transition_count": len(vis_transitions),
            "by_type": {
                t: sum(1 for n in vis_nodes if n["type"] == t)
                for t in _TYPE_COLOR
                if any(n["type"] == t for n in vis_nodes)
            },
            "communities":        communities,
            "type_order":         [t for t in _TYPE_ORDER
                                   if any(n["type"] == t for n in vis_nodes)],
            # Why the graph may be empty — the page's only data path.
            "processed_messages": len(graph._processed_hashes),
            "load_failed":        graph._load_failed,
            "persistence_broken": graph._persistence_broken,
            "data_loss_detected": graph._data_loss_detected,
        },
    }


def to_obsidian_canvas(graph: "GraphMemory") -> dict:
    """
    Export graph as Obsidian Canvas JSON (.canvas format).
    Nodes are laid out in clusters by type using a grid algorithm.
    Save the output as a .canvas file and open in Obsidian.
    """
    canvas_nodes = []
    canvas_edges = []
    active_ids:  set[str] = set()

    by_type: dict[str, list] = {}
    for nid, n in graph._nodes.items():
        if n._evicted:
            continue
        t = n.type.value
        by_type.setdefault(t, []).append((nid, n))
        active_ids.add(nid)

    node_positions: dict[str, tuple[float, float]] = {}

    for node_type, node_list in by_type.items():
        cx, cy = _CLUSTER_CENTERS.get(node_type, (1500, 1000))
        count = len(node_list)
        radius = max(120, min(300, count * 40))
        for i, (nid, n) in enumerate(node_list):
            x, y = (cx, cy) if count == 1 else (
                cx + radius * math.cos(2 * math.pi * i / count),
                cy + radius * math.sin(2 * math.pi * i / count),
            )
            label     = n.label[:80]
            sum_line  = f"\n> {n.summary[:100]}" if n.summary else ""
            stat_line = f"\n**Status:** {n.status.value}"
            conf_line = f"  **Confidence:** {n.confidence:.0%}"

            canvas_nodes.append({
                "id":     nid[:16],
                "type":   "text",
                "x":      round(x - 140),
                "y":      round(y - 40),
                "width":  280,
                "height": 90 + (40 if n.summary else 0),
                "color":  _TYPE_COLOR_OBS.get(node_type, ""),
                "text":   f"**[{node_type.upper()}]** {label}{sum_line}{stat_line}{conf_line}",
            })
            node_positions[nid] = (x, y)

    for i, e in enumerate(graph._edges):
        if e.source_id not in active_ids or e.target_id not in active_ids:
            continue
        canvas_edges.append({
            "id":       f"edge-{i}",
            "fromNode": e.source_id[:16],
            "fromSide": "right",
            "toNode":   e.target_id[:16],
            "toSide":   "left",
            "label":    _EDGE_LABEL.get(e.type.value, e.type.value),
            "color":    "4" if e.type.value == "supersedes" else "",
        })

    legend = [
        "## TokenMizer Graph",
        f"**Session:** `{graph.session_id}`",
        f"**Nodes:** {len(canvas_nodes)}  **Edges:** {len(canvas_edges)}",
        "",
        "**Node types:**",
        "GOAL  |  DECISION  |  TASK",
        "FILE  |  ERROR  |  ENDPOINT",
    ]
    canvas_nodes.insert(0, {
        "id": "legend", "type": "text",
        "x": -300, "y": 0, "width": 240, "height": 200,
        "color": "6", "text": "\n".join(legend),
    })

    return {"nodes": canvas_nodes, "edges": canvas_edges}


# ── Shareable standalone HTML ─────────────────────────────────────────────────
#
# Zero external dependencies by design: the artifact must render offline
# and in network-restricted environments, so the force layout is inlined
# rather than loaded from a CDN (a test forbids external scripts). What
# the page shows: nodes filled by detected community (communities.py)
# with a thin ring in the node-type color; a side panel listing the
# communities with counts and checkboxes, a click-to-inspect node detail
# with its neighbors, and the decision-supersession history (arcs on the
# canvas, a clickable timeline in the panel); search, active-only filter,
# fit-to-view, wheel zoom/pan, drag-to-pin, PNG export. An empty graph
# gets an explanation of why, driven by the health fields in meta,
# instead of a blank canvas. Theme variables match the dashboard's.

# Authored as a RAW string: the page carries JavaScript escapes
# (\u2026 in a label, \w in a regex) that a normal Python string would
# consume before the browser ever saw them.
_SHARE_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark"><head><meta charset="utf-8">
<title>TokenMizer — __SESSION__</title>
<style>
 :root{--bg:#0b0d14;--surface:#12141c;--surface2:#171a24;--border:#242838;--line:#1c2030;
       --accent:#9085e9;--accent2:#5ee7c8;--text:#e9ecf5;--muted:#868da3;--faint:#5b6176;
       --good:#199e70;--bad:#e66767;--stroke:#0b0d14;--side:380px;
       --shadow:0 10px 34px #0009}
 html[data-theme="light"]{--bg:#f7f8fc;--surface:#ffffff;--surface2:#f1f3f9;--border:#dde1ed;
       --line:#e6e9f2;--accent:#4a3aa7;--accent2:#1baf7a;--text:#12141c;--muted:#5d6478;
       --faint:#8c93a8;--stroke:#ffffff;--shadow:0 10px 34px #12141c18}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--text);overflow:hidden;
      font:14px/1.45 'Inter',system-ui,-apple-system,sans-serif;
      transition:background .18s,color .18s}
 #hdr{position:fixed;top:0;left:0;right:var(--side);padding:14px 22px;display:flex;gap:9px;
      align-items:center;flex-wrap:wrap;z-index:3;pointer-events:none}
 #hdr>*{pointer-events:auto}
 #hdr b{font-size:17px;font-weight:650;letter-spacing:-.015em;margin-right:4px}
 .stat{color:var(--muted);font-size:12px;white-space:nowrap}
 .stat i{color:var(--accent2);font-style:normal;font-weight:650}
 #search{background:var(--surface);border:1px solid var(--border);color:var(--text);
      border-radius:9px;padding:6px 11px;font:inherit;font-size:12px;width:165px;
      outline:none;margin-left:auto}
 #search:focus{border-color:var(--accent)}
 .btn{background:var(--surface);border:1px solid var(--border);color:var(--text);
      border-radius:9px;padding:6px 11px;font-size:12px;cursor:pointer;user-select:none;
      white-space:nowrap;transition:border-color .15s,color .15s,background .15s}
 .btn:hover{border-color:var(--accent)}
 .btn.on{border-color:var(--accent2);color:var(--accent2)}
 .seg{display:flex;border:1px solid var(--border);border-radius:9px;overflow:hidden;
      background:var(--surface)}
 .seg .btn{border:0;border-radius:0;padding:6px 13px}
 .seg .btn.on{background:var(--accent);color:#fff}

 html.nopanel{--side:0px} html.nopanel #side{display:none}
 #side{position:fixed;top:0;right:0;bottom:0;width:var(--side);background:var(--surface);
      border-left:1px solid var(--border);z-index:4;overflow-y:auto;padding:22px 22px 30px}
 #side h1{font-size:19px;font-weight:650;letter-spacing:-.015em;margin:0 0 4px}
 #side .sub{color:var(--muted);font-size:12.5px;margin-bottom:20px;line-height:1.5}
 #side h2{font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);
      margin:22px 0 9px;display:flex;justify-content:space-between;align-items:baseline}
 #side h2 em{font-style:normal;text-transform:none;letter-spacing:0;color:var(--faint)}
 .kpi{display:flex;justify-content:space-between;align-items:baseline;
      padding:7px 0;border-bottom:1px solid var(--line);font-size:13px}
 .kpi:last-child{border-bottom:none}
 .kpi span{color:var(--muted)}
 .kpi b{font-weight:650;font-variant-numeric:tabular-nums;font-size:15px}
 .kpi b.good{color:var(--good)}.kpi b.bad{color:var(--bad)}
 .row{display:flex;align-items:center;gap:9px;padding:5px 5px;border-radius:7px;
      cursor:pointer;font-size:13px;user-select:none}
 .row:hover{background:var(--surface2)}
 .row .dot{width:9px;height:9px;border-radius:50%;flex:none}
 .row .name{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
 .row .n{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
 .row.off .name,.row.off .dot,.row.off .n{opacity:.3}
 .row input{accent-color:var(--accent);margin:0;flex:none}
 .hot{display:flex;justify-content:space-between;gap:10px;align-items:baseline;
      padding:6px 0;border-bottom:1px solid var(--line);font-size:12.5px;cursor:pointer}
 .hot:last-child{border-bottom:none}
 .hot:hover .hl{color:var(--accent2)}
 .hot .hl{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
 .hot .hn{color:var(--muted);font-size:12px;white-space:nowrap;font-variant-numeric:tabular-nums}
 .flow{display:grid;grid-template-columns:auto 1fr 2.4em;gap:9px;align-items:center;
       padding:4px 0;font-size:12.5px}
 .flow .fl{color:var(--muted);white-space:nowrap}
 .flow .ft{height:8px;border-radius:0 4px 4px 0;min-width:2px}
 .flow .fn{color:var(--muted);text-align:right;font-variant-numeric:tabular-nums;font-size:12px}
 #detail{display:none}
 #detail .lbl{font-size:14px;font-weight:600;margin-bottom:8px;word-break:break-word;line-height:1.35}
 .pill{display:inline-block;font-size:11px;padding:1px 9px;border-radius:10px;
       border:1px solid;margin:0 6px 6px 0}
 .kv{color:var(--muted);font-size:12px;margin:3px 0}.kv b{color:var(--text);font-weight:500}
 .bar{height:4px;border-radius:2px;background:var(--line);margin:2px 0 8px;overflow:hidden}
 .bar i{display:block;height:100%;border-radius:2px}
 .neighbors{list-style:none;margin:6px 0 0;padding:0}
 .neighbors li{padding:4px 5px;border-radius:6px;cursor:pointer;font-size:12px;
      display:flex;gap:7px;align-items:center}
 .neighbors li:hover{background:var(--surface2)}
 .neighbors li .name{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1}
 .neighbors li .rel{color:var(--faint);font-size:10px;text-transform:uppercase;
      letter-spacing:.05em;flex:none}
 .chain{border-left:2px solid var(--bad);margin:8px 0 0 3px;padding:2px 0 2px 10px}
 .chain .hop{font-size:12px;margin-bottom:7px}
 .chain .hop .old{color:var(--muted);text-decoration:line-through}
 .chain .hop .new{font-weight:600}
 .chain .hop .why{color:var(--muted);font-size:11.5px;margin-top:2px}
 .tr-item{border:1px solid var(--border);border-radius:9px;padding:10px 12px;margin-bottom:9px;
      cursor:pointer;transition:border-color .15s}
 .tr-item:hover,.tr-item.sel{border-color:var(--bad)}
 .tr-item .old{color:var(--muted);text-decoration:line-through;font-size:12px}
 .tr-item .arrow{color:var(--bad);margin:0 4px}
 .tr-item .new{font-size:13px;font-weight:600}
 .tr-item .why{color:var(--muted);font-size:12px;margin-top:5px}
 .tr-item .meta{color:var(--faint);font-size:11px;margin-top:4px}
 .empty{color:var(--muted);font-size:12.5px;border:1px dashed var(--border);border-radius:9px;
      padding:15px;text-align:center;line-height:1.5}
 #emptyCard{position:fixed;top:50%;left:calc((100vw - var(--side)) / 2);transform:translate(-50%,-50%);
      width:min(560px,80vw);background:var(--surface);border:1px solid var(--border);
      border-radius:14px;padding:24px 26px;z-index:3;display:none;box-shadow:var(--shadow)}
 #emptyCard h3{margin:0 0 10px;font-size:17px}
 #emptyCard p{color:var(--muted);font-size:13px;margin:9px 0;line-height:1.6}
 #emptyCard code{background:var(--surface2);padding:1px 6px;border-radius:4px;font-size:12px;color:var(--text)}
 #tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--border);
      border-radius:10px;padding:10px 12px;font-size:12px;max-width:330px;z-index:9;display:none;
      box-shadow:var(--shadow);line-height:1.45}
 #tip b{color:var(--text)}#tip .st{color:var(--accent2)}
 #ftr{position:fixed;bottom:14px;left:22px;color:var(--faint);font-size:12px;z-index:3}
 #ftr a{color:var(--accent);text-decoration:none}
 svg{cursor:grab;display:block}svg:active{cursor:grabbing}
 .lane{fill:var(--faint);font-size:10.5px;letter-spacing:.07em;text-transform:uppercase}
 .axis{stroke:var(--border);stroke-width:1}
 .axtx{fill:var(--faint);font-size:10.5px}
</style></head><body>
<div id="hdr"><b>__SESSION__</b>
 <span class="stat"><i>__NODES__</i> nodes</span>
 <span class="stat"><i>__EDGES__</i> relations</span>
 <span class="stat"><i>__DECISIONS__</i> decisions</span>
 <span class="stat"><i>__TRANSITIONS__</i> changed</span>
 <input id="search" type="search" placeholder="Search nodes"/>
 <span class="seg"><span class="btn on" id="viewRadial">Radial</span><span class="btn" id="viewGraph">Force</span><span class="btn" id="viewTime">Timeline</span></span>
 <span class="btn" id="activeOnly">Active only</span>
 <span class="btn" id="fit">Fit</span>
 <span class="btn" id="theme">Light</span>
 <span class="btn" id="exportPng">PNG</span>
</div>
<div id="emptyCard"><h3>No nodes yet</h3><div id="emptyBody"></div></div>
<div id="side">
 <h1>Session memory</h1>
 <div class="sub">Extracted from the transcript on every turn &mdash; not hand-drawn.</div>
 <div id="kpis"></div>
 <h2>Node types <em id="typeTotal"></em></h2>
 <div id="typeList"></div>
 <h2>Communities <em id="comTotal"></em></h2>
 <label class="row" id="selectAllRow"><input type="checkbox" id="selectAll" checked/>
  <span class="name">Select all</span></label>
 <div id="comList"></div>
 <h2>Hotspots <em>what the session hangs off</em></h2>
 <div id="hotList"></div>
 <h2>Relations <em>what connects to what</em></h2>
 <div id="flowList"></div>
 <div id="detail"><h2>Node</h2><div id="detailBody"></div></div>
 <h2>Decision history</h2>
 <div id="trList"></div>
 <h2>Reading the picture</h2>
 <div id="statusLegend"></div>
</div>
<div id="ftr">session memory graph &middot; <a href="https://github.com/Shweta-Mishra-ai/tokenmizer">TokenMizer</a> &middot; pip install tokenmizer</div>
<div id="tip"></div>
<script>
"use strict";
const DATA=__DATA__, COLOR_DARK=__COLORS__, COLOR_LIGHT=__COLORS_LIGHT__;
const META=DATA.meta||{};
let COLOR=COLOR_DARK;
const INACTIVE=new Set(["superseded","archived","invalidated","modified"]);
// Embedded small — the dashboard drops this page into an iframe — the panel
// would take more room than the picture it explains, so a narrow page is
// the graph alone. A preview frame gets less benefit of the doubt than a
// real window, because it is a picture first and a reading surface second.
const EMBEDDED=(()=>{try{return window.top!==window.self}catch(e){return true}})();
// A small frame cannot carry node names: zoom-to-fit would shrink them to a
// few illegible pixels. It shows the shape of the session instead — arcs,
// chords and type names — and the full page carries the reading.
function previewSize(){return EMBEDDED&&((innerWidth||1280)<1280||(innerHeight||800)<700)}
let PREVIEW=previewSize();
function sideW(){return PREVIEW||(innerWidth||1280)<900?0:380}
let SIDE=sideW();
function applySide(){document.documentElement.classList.toggle("nopanel",!SIDE)}
applySide();
let W=Math.max(360,(innerWidth||1280)-SIDE), H=Math.max(300,innerHeight||800);
const NS="http://www.w3.org/2000/svg";
function el(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function $(id){return document.getElementById(id)}
function when(ts){return ts?new Date(ts*1000).toLocaleString():""}
function colorOf(t){return COLOR[t]||"#8a8f9e"}

const svg=el("svg",{width:W,height:H,"font-family":"Inter,system-ui,sans-serif"},document.body);
addEventListener("resize",()=>{PREVIEW=previewSize();SIDE=sideW();applySide();W=Math.max(360,innerWidth-SIDE);H=Math.max(300,innerHeight);
  svg.setAttribute("width",W);svg.setAttribute("height",H);
  // A layout computed for one canvas is not framed for another, and a
  // resize is the one moment the reader is looking at the whole picture.
  // A filter toggle deliberately does not refit: it would move everything
  // the reader was comparing.
  relayout();fit()});
const defs=el("defs",{},svg);
const mk=el("marker",{id:"arr",viewBox:"0 -5 10 10",refX:16,refY:0,markerWidth:6,markerHeight:6,orient:"auto"},defs);
el("path",{d:"M0,-5L10,0L0,5",fill:"#e66767"},mk);
const vp=el("g",{},svg);
const gH=el("g",{},vp), gAx=el("g",{},vp), gE=el("g",{},vp),
      gT=el("g",{},vp), gL=el("g",{},vp), gN=el("g",{},vp);

// ---- data ------------------------------------------------------------------
const nodes=DATA.nodes, edges=DATA.edges, trans=DATA.transitions||[];
const byId={}; nodes.forEach(n=>byId[n.id]=n);
const COMS={}; (META.communities||[]).forEach(c=>COMS[c.id]=c);
function comOf(n){return COMS[n.community]||{id:-1,name:"Unclustered",color:"#8a8f9e"}}
const links=edges.map(e=>({s:byId[e.source],t:byId[e.target],type:e.type}))
                 .filter(l=>l.s&&l.t);
const tlinks=trans.map(t=>({s:byId[t.from_id],t:byId[t.to_id],tr:t}))
                  .filter(l=>l.s&&l.t);
const nbrs={}; nodes.forEach(n=>nbrs[n.id]=[]);
links.forEach(l=>{nbrs[l.s.id].push({n:l.t,rel:l.type,dir:"out"});
                  nbrs[l.t.id].push({n:l.s,rel:l.type,dir:"in"})});
const trByNode={}; trans.forEach(t=>{(trByNode[t.from_id]=trByNode[t.from_id]||[]).push(t);
                                     (trByNode[t.to_id]=trByNode[t.to_id]||[]).push(t)});
// Server-side order: types in reading order, heaviest first inside a type.
const typesPresent=(META.type_order||[]).filter(t=>nodes.some(n=>n.type===t));

let seed=1;
function rand(){seed=(seed*1103515245+12345)%2147483648;return seed/2147483648}
const comIds=Object.keys(COMS).map(Number).sort((a,b)=>a-b);
nodes.forEach(n=>{
  const a=2*Math.PI*Math.max(0,comIds.indexOf(n.community))/Math.max(comIds.length,1);
  const r=nbrs[n.id].length?260:380;
  n.x=W/2+Math.cos(a)*r+(rand()-0.5)*140;
  n.y=H/2+Math.sin(a)*r+(rand()-0.5)*140;
  n.vx=0;n.vy=0;n.fx=null;n.fy=null;
  n._lw=Math.min(n.label.length,34)*6.2+(n.size||10)+10;
  // Drawn radius: type weight, nudged by the node's own importance, so a
  // pivotal decision reads larger than a passing one of the same type.
  n._r=Math.max(3.2,(n.size||10)*0.42+(n.importance||0.5)*4.5);
});

// ---- radial layout ---------------------------------------------------------
// The default view, and the reason the colour palette is legal: a node's
// type is read off WHICH ARC it sits on, with the type named beside the
// arc, before colour is consulted at all. See the palette note in
// visualization.py before changing any of this.
const GROUP_GAP=0.055;           // radians of blank between two type arcs
// A type with one node would otherwise get a zero-length arc and nowhere
// to hang its name, which is the label the whole encoding rests on.
const MIN_ARC=0.10;
// How far a node's label reaches outward from its dot. Both the type ring
// and the zoom-to-fit depend on this, so there is one estimate, not two.
function labelExtent(n){return PREVIEW?n._r+6:n._r+14+Math.min(n.label.length,26)*5.9}
function layoutRadial(){
  const vis=nodes.filter(visible);
  gAx.innerHTML="";
  if(!vis.length)return;
  const groups=typesPresent.map(t=>({type:t,items:vis.filter(n=>n.type===t)}))
                           .filter(g=>g.items.length);
  if(!groups.length)return;
  const cx=W/2, cy=H/2+8;
  // Zoom-to-fit rescales everything afterwards, so what R decides is not the
  // size of the circle but how much of it the labels eat: keep the dot ring
  // comfortably wider than the longest label so the type ring hugs the text.
  const maxExt=vis.reduce((m,n)=>Math.max(m,labelExtent(n)),0);
  const R=Math.max(220,maxExt*2.6,Math.min(W,H)*0.30);
  const total=vis.length;
  // Share the circle by node count, then lift every group to MIN_ARC and
  // take the difference back off the groups big enough to spare it.
  const free=2*Math.PI-GROUP_GAP*groups.length;
  const raw=groups.map(g=>free*(g.items.length/total));
  const lifted=raw.map(a=>Math.max(a,MIN_ARC));
  const over=lifted.reduce((s,a)=>s+a,0)-free;
  if(over>0){
    const slack=lifted.reduce((s,a)=>s+Math.max(0,a-MIN_ARC),0);
    if(slack>0)for(let i=0;i<lifted.length;i++)
      lifted[i]-=Math.max(0,lifted[i]-MIN_ARC)/slack*over;
  }
  let angle=-Math.PI/2+GROUP_GAP/2;   // start at 12 o'clock
  const bands=[];
  groups.forEach((g,gi)=>{
    const arc=lifted[gi];
    // One item would sit on the arc's leading edge; centre it instead.
    const step=g.items.length>1?arc/(g.items.length-1):0;
    const start=g.items.length>1?angle:angle+arc/2;
    g.items.forEach((n,i)=>{
      const a=start+i*step;
      n._a=a; n.fx=cx+Math.cos(a)*R; n.fy=cy+Math.sin(a)*R;
      n.x=n.fx; n.y=n.fy;
    });
    bands.push({type:g.type,a0:angle-GROUP_GAP*0.30,a1:angle+arc+GROUP_GAP*0.30});
    angle+=arc+GROUP_GAP;
  });
  // The ring of type names sits OUTSIDE every node label, so the picture
  // reads dot, then the thing's own name, then the name of its layer. Put
  // the ring any closer and the two kinds of text collide.
  const ar=R+maxExt+20;
  bands.forEach((b,gi)=>{
    // Text on a path follows the path's direction, so an arc on the bottom
    // half is drawn backwards or its name comes out upside down.
    const down=Math.sin((b.a0+b.a1)/2)>0, span=Math.abs(b.a1-b.a0);
    const p0=down?b.a1:b.a0, p1=down?b.a0:b.a1;
    const id="arc"+gi;
    el("path",{id:id,d:"M"+(cx+Math.cos(p0)*ar)+","+(cy+Math.sin(p0)*ar)+
        " A"+ar+","+ar+" 0 "+(span>Math.PI?1:0)+" "+(down?0:1)+" "+
        (cx+Math.cos(p1)*ar)+","+(cy+Math.sin(p1)*ar),
      fill:"none",stroke:colorOf(b.type),"stroke-opacity":.5,"stroke-width":2.5,
      "stroke-linecap":"round"},gAx);
    // A name wider than its own arc would run over the neighbouring type,
    // so a short arc keeps the colour band and leaves the name to the legend.
    if(ar*span<b.type.length*7.6+14)return;
    const t=el("text",{class:"lane",fill:colorOf(b.type),dy:down?15:-8},gAx);
    const tp=document.createElementNS(NS,"textPath");
    tp.setAttribute("href","#"+id);
    tp.setAttributeNS("http://www.w3.org/1999/xlink","xlink:href","#"+id);
    tp.setAttribute("startOffset","50%");
    tp.setAttribute("text-anchor","middle");
    tp.textContent=b.type;
    t.appendChild(tp);
  });
  radialCentre={cx:cx,cy:cy,R:R,band:ar};
}
let radialCentre={cx:0,cy:0,R:1,band:0};

// A chord between two points on the circle, bowed toward the centre by how
// far apart they are: neighbours on the same arc keep a shallow curve,
// opposite sides pass near the middle. This is what turns a hairball into
// a readable pattern.
function chord(a,b){
  if(mode!=="radial"||a._a===undefined||b._a===undefined)
    return "M"+a.x+","+a.y+" L"+b.x+","+b.y;
  const {cx,cy,R}=radialCentre;
  let d=Math.abs(a._a-b._a); if(d>Math.PI)d=2*Math.PI-d;
  const pull=0.08+0.62*(1-d/Math.PI);
  let m=(a._a+b._a)/2;
  if(Math.abs(a._a-b._a)>Math.PI)m+=Math.PI;
  const kr=R*pull;
  return "M"+a.x+","+a.y+" Q"+(cx+Math.cos(m)*kr)+","+(cy+Math.sin(m)*kr)+" "+b.x+","+b.y;
}

// ---- force simulation ------------------------------------------------------
let alpha=1, mode="radial";
const centroids={};
function step(){
  if(mode!=="graph")return;
  for(const k in centroids)delete centroids[k];
  nodes.forEach(n=>{const c=centroids[n.community]||(centroids[n.community]={x:0,y:0,k:0});
                    c.x+=n.x;c.y+=n.y;c.k++});
  for(const k in centroids){centroids[k].x/=centroids[k].k;centroids[k].y/=centroids[k].k}
  for(let i=0;i<nodes.length;i++){
    const a=nodes[i];
    for(let j=i+1;j<nodes.length;j++){
      const b=nodes[j];
      let dx=b.x-a.x,dy=b.y-a.y,d2=dx*dx+dy*dy||1,d=Math.sqrt(d2);
      let f=Math.min(6000/d2,6)*alpha;
      const minD=(a.size||10)+(b.size||10)+16;
      if(d<minD)f+=(minD-d)*0.3;
      const lw=(a._lw+b._lw)/2, ady=Math.abs(dy), adx=Math.abs(dx);
      if(ady<24&&adx<lw){const push=(lw-adx)*0.04*alpha, sx=dx<0?-1:1; a.vx-=sx*push; b.vx+=sx*push;}
      dx/=d;dy/=d; a.vx-=dx*f;a.vy-=dy*f; b.vx+=dx*f;b.vy+=dy*f;
    }
    const c=centroids[a.community];
    if(c&&c.k>1&&nbrs[a.id].length){a.vx+=(c.x-a.x)*0.006*alpha;a.vy+=(c.y-a.y)*0.006*alpha}
    a.vx+=(W/2-a.x)*0.0012*alpha; a.vy+=(H/2-a.y)*0.0012*alpha;
  }
  links.forEach(l=>{
    let dx=l.t.x-l.s.x,dy=l.t.y-l.s.y,d=Math.sqrt(dx*dx+dy*dy)||1;
    const f=(d-140)*0.012*alpha; dx/=d;dy/=d;
    l.s.vx+=dx*f*d*0.02;l.s.vy+=dy*f*d*0.02;
    l.t.vx-=dx*f*d*0.02;l.t.vy-=dy*f*d*0.02;
  });
  nodes.forEach(n=>{
    if(n.fx!==null){n.x=n.fx;n.y=n.fy;n.vx=0;n.vy=0;return}
    n.vx*=0.8;n.vy*=0.8; n.x+=n.vx;n.y+=n.vy;
  });
  alpha*=0.97;
}
function tick(){step();render();if(mode==="graph"&&alpha>0.003)requestAnimationFrame(tick)}
function reheat(a){if(mode!=="graph")return;alpha=Math.max(alpha,a);requestAnimationFrame(tick)}

// ---- timeline layout -------------------------------------------------------
function layoutTimeline(){
  const vis=nodes.filter(visible);
  gAx.innerHTML="";
  if(!vis.length)return;
  const times=vis.map(n=>n.created_at||n.valid_from||0).filter(t=>t>0);
  let t0=times.length?Math.min.apply(null,times):0;
  let t1=times.length?Math.max.apply(null,times):0;
  // A whole transcript checkpointed in one call gives every node the same
  // created_at, and a time axis over a span of zero is a lie drawn to four
  // decimal places. The order nodes entered the graph IS the order the
  // transcript stated them, so fall back to that and say so. The bar is a
  // minute: below that every tick renders the same clock time.
  const byTime=(t1-t0)>=60;
  if(!byTime){t0=0;t1=Math.max(1,vis.length-1)}
  const ordinal={}; nodes.forEach((n,i)=>ordinal[n.id]=i);
  const posOf=n=>byTime?(n.created_at||n.valid_from||t0):ordinal[n.id];
  const lanes=typesPresent.filter(t=>vis.some(n=>n.type===t));
  const busiest=Math.max.apply(null,
    lanes.map(t=>vis.filter(n=>n.type===t).length).concat([1]));
  const laneH=Math.max(64,Math.min(150,(H-170)/Math.max(lanes.length,1),
                                   28+Math.min(busiest,6)*17));
  const x0=150, x1=Math.max(x0+320,W-230);
  // Lanes start below the axis caption and its tick labels; any higher and
  // the first lane's dots sit on top of the "earliest" tick.
  const LABEL_W=210, ROW_H=17, LANE_Y0=130;
  const rows={};
  vis.slice().sort((a,b)=>posOf(a)-posOf(b)).forEach(n=>{
    const li=lanes.indexOf(n.type);
    const frac=t1>t0?(posOf(n)-t0)/(t1-t0):0.5;
    const x=x0+frac*(x1-x0);
    const lane=rows[n.type]||(rows[n.type]=[]);
    let row=lane.findIndex(lastRight=>x>=lastRight);
    if(row===-1){row=lane.length;lane.push(0)}
    lane[row]=x+LABEL_W;
    n._a=undefined;
    n.fx=x; n.fy=LANE_Y0+li*laneH+row*ROW_H-(laneH/2-14);
    n.x=n.fx;n.y=n.fy;
  });
  lanes.forEach((t,i)=>{
    const y=LANE_Y0+i*laneH;
    el("line",{x1:x0-38,y1:y,x2:x1+40,y2:y,class:"axis","stroke-opacity":.45},gAx);
    const lab=el("text",{x:x0-46,y:y+4,"text-anchor":"end",class:"lane",fill:colorOf(t)},gAx);
    lab.textContent=t;
  });
  if(t1>t0){
    for(let i=0;i<=4;i++){
      const x=x0+(x1-x0)*i/4, v=t0+(t1-t0)*i/4;
      el("line",{x1:x,y1:72,x2:x,y2:LANE_Y0+lanes.length*laneH-laneH+40,class:"axis","stroke-opacity":.16},gAx);
      const tx=el("text",{x:x,y:64,"text-anchor":"middle",class:"axtx"},gAx);
      tx.textContent=byTime
        ? new Date(v*1000).toLocaleString(undefined,
            {month:"short",day:"numeric",hour:"2-digit",minute:"2-digit"})
        : (i===0?"earliest":i===4?"latest":"");
    }
    const cap=el("text",{x:x0,y:44,class:"axtx"},gAx);
    cap.textContent=byTime?"when each fact entered the session"
                          :"the order the session stated each fact (all checkpointed together)";
  }
}

function relayout(){
  if(mode==="radial")layoutRadial();
  else if(mode==="timeline")layoutTimeline();
  render();
}
function setMode(m){
  mode=m;
  $("viewRadial").classList.toggle("on",m==="radial");
  $("viewGraph").classList.toggle("on",m==="graph");
  $("viewTime").classList.toggle("on",m==="timeline");
  gH.style.display=m==="graph"?"":"none";
  if(m==="graph"){
    gAx.innerHTML="";
    nodes.forEach(n=>{n.fx=null;n.fy=null;n._a=undefined});
    alpha=0.6;requestAnimationFrame(tick);setTimeout(fit,260);
  }else{relayout();fit()}
}

// ---- render ----------------------------------------------------------------
// Edges are paths, not lines: in radial mode they are chords bowed through
// the centre, and a path degrades to a straight segment everywhere else.
const eEls=links.map(l=>el("path",{fill:"none",stroke:colorOf(l.s.type),
  "stroke-opacity":0.34,"stroke-width":1.25,"stroke-linecap":"round"},gE));
const eLabels=links.map(()=>{const t=el("text",{class:"axtx","text-anchor":"middle"},gL);
  t.style.display="none";return t});
const tEls=tlinks.map(l=>{
  const p=el("path",{fill:"none",stroke:"#e66767","stroke-width":1.7,
    "stroke-dasharray":"6 4","stroke-opacity":0.9,"marker-end":"url(#arr)"},gT);
  p.dataset.trid=l.tr.id; return p;
});
const hullEls={};
comIds.forEach(id=>{hullEls[id]=el("path",{fill:COMS[id].color,"fill-opacity":0.07,
  stroke:COMS[id].color,"stroke-opacity":0.22,"stroke-width":1.5,"stroke-linejoin":"round"},gH)});

const nEls=nodes.map(n=>{
  const g=el("g",{cursor:"pointer"},gN);
  const inactive=INACTIVE.has(n.status);
  const halo=el("circle",{r:n._r+5,"fill-opacity":0.16},g);
  const disc=el("circle",{r:n._r,"fill-opacity":inactive?0.3:1},g);
  const ring=el("circle",{r:n._r+2.4,fill:"none","stroke-width":1.4},g);
  if(inactive)
    el("circle",{r:n._r+5.5,fill:"none",stroke:"#8a8f9e","stroke-dasharray":"2.5 2.5",
      "stroke-opacity":0.6,"stroke-width":1},g);
  if(n.status==="invalidated")
    el("circle",{r:n._r+5.5,fill:"none",stroke:"#e66767","stroke-dasharray":"2 2","stroke-width":1.3},g);
  if(n.status==="contested")
    el("circle",{r:n._r+5.5,fill:"none",stroke:"#c98500","stroke-dasharray":"1 3","stroke-width":1.5},g);
  const lbl=el("text",{"font-size":10.5,"paint-order":"stroke",
    "stroke-width":3.2,"stroke-linejoin":"round","dominant-baseline":"middle"},g);
  lbl.textContent=n.label.length>26?n.label.slice(0,25)+"…":n.label;
  if(PREVIEW)lbl.setAttribute("display","none");
  if(inactive)lbl.setAttribute("text-decoration","line-through");
  n._halo=halo;n._disc=disc;n._ring=ring;n._lbl=lbl;
  g._n=n; n._el=g;
  let drag=false, moved=false;
  g.addEventListener("pointerdown",ev=>{drag=true;moved=false;g.setPointerCapture(ev.pointerId);ev.stopPropagation()});
  g.addEventListener("pointermove",ev=>{if(!drag||mode!=="graph")return;
    const m=pt(ev);n.fx=m.x;n.fy=m.y;n.x=m.x;n.y=m.y;moved=true;reheat(0.3)});
  g.addEventListener("pointerup",()=>{drag=false;if(!moved)select(n)});
  g.addEventListener("dblclick",ev=>{ev.stopPropagation();
    if(mode==="graph"){n.fx=null;n.fy=null;reheat(0.3)}});
  g.addEventListener("pointerenter",ev=>{
    hover=n;applyEmphasis();
    tip.style.display="block";moveTip(ev);
    const changed=(trByNode[n.id]||[]).length, deg=nbrs[n.id].length;
    tip.innerHTML="<b>"+esc(n.full_label||n.label)+"</b><br><span class='st'>"+esc(n.type)+
      " &middot; "+esc(n.status)+"</span>"+(n.summary?"<br>"+esc(n.summary):"")+
      "<br><span style='color:var(--faint)'>"+esc(comOf(n).name)+
      " &middot; "+deg+" relation"+(deg===1?"":"s")+
      (changed?" &middot; "+changed+" change"+(changed>1?"":""):"")+"</span>";
  });
  g.addEventListener("pointermove",moveTip);
  g.addEventListener("pointerleave",()=>{hover=null;applyEmphasis();tip.style.display="none"});
  return g;
});
const tip=$("tip");
function moveTip(ev){
  const w=340, x=Math.min(ev.clientX+14, innerWidth-SIDE-w);
  tip.style.left=Math.max(8,x)+"px";
  tip.style.top=Math.min(ev.clientY+10, innerHeight-150)+"px";
}

function hull(pts,pad){
  if(pts.length<3)return null;
  const p=pts.slice().sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
  const cross=(o,a,b)=>(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);
  const lo=[],up=[];
  for(const q of p){while(lo.length>=2&&cross(lo[lo.length-2],lo[lo.length-1],q)<=0)lo.pop();lo.push(q)}
  for(let i=p.length-1;i>=0;i--){const q=p[i];
    while(up.length>=2&&cross(up[up.length-2],up[up.length-1],q)<=0)up.pop();up.push(q)}
  lo.pop();up.pop();
  const h=lo.concat(up);
  if(h.length<3)return null;
  const cx=h.reduce((s,q)=>s+q[0],0)/h.length, cy=h.reduce((s,q)=>s+q[1],0)/h.length;
  const out=h.map(([x,y])=>{const dx=x-cx,dy=y-cy,d=Math.hypot(dx,dy)||1;
    return [x+dx/d*pad, y+dy/d*pad]});
  let d="M"+out[0][0]+","+out[0][1];
  for(let i=0;i<out.length;i++){
    const a=out[i], b=out[(i+1)%out.length];
    d+=" Q"+a[0]+","+a[1]+" "+((a[0]+b[0])/2)+","+((a[1]+b[1])/2);
  }
  return d+"Z";
}
function renderHulls(){
  if(mode!=="graph"){comIds.forEach(id=>hullEls[id].style.display="none");return}
  comIds.forEach(id=>{
    const c=COMS[id], path=hullEls[id];
    const pts=nodes.filter(n=>n.community===id&&visible(n)&&nbrs[n.id].length)
                   .map(n=>[n.x,n.y]);
    const d=(c.name==="Unclustered")?null:hull(pts,34);
    if(d){path.setAttribute("d",d);path.style.display=""}else path.style.display="none";
  });
}
function render(){
  links.forEach((l,i)=>{
    eEls[i].setAttribute("d",chord(l.s,l.t));
    const lab=eLabels[i];
    if(lab.style.display!=="none"){
      lab.setAttribute("x",(l.s.x+l.t.x)/2);lab.setAttribute("y",(l.s.y+l.t.y)/2-5);
    }});
  tlinks.forEach((l,i)=>{
    if(mode==="radial"){tEls[i].setAttribute("d",chord(l.s,l.t));return}
    const mx=(l.s.x+l.t.x)/2, my=(l.s.y+l.t.y)/2;
    const dx=l.t.x-l.s.x, dy=l.t.y-l.s.y, d=Math.sqrt(dx*dx+dy*dy)||1;
    tEls[i].setAttribute("d","M"+l.s.x+","+l.s.y+" Q"+(mx-dy/d*40)+","+(my+dx/d*40)+" "+l.t.x+","+l.t.y)});
  nodes.forEach(n=>{
    n._el.setAttribute("transform","translate("+n.x+","+n.y+")");
    // In radial mode a label reads outward along its own spoke; elsewhere
    // it sits to the right of the dot.
    if(mode==="radial"&&n._a!==undefined){
      const flip=Math.cos(n._a)<0, off=n._r+7;
      n._lbl.setAttribute("text-anchor",flip?"end":"start");
      n._lbl.setAttribute("x",flip?-off:off);
      n._lbl.setAttribute("y",0);
      n._lbl.setAttribute("transform","rotate("+(n._a*180/Math.PI+(flip?180:0))+")");
    }else{
      n._lbl.setAttribute("text-anchor","start");
      n._lbl.setAttribute("x",n._r+6);n._lbl.setAttribute("y",0);
      n._lbl.removeAttribute("transform");
    }
  });
  renderHulls();
}

// ---- zoom / pan / fit ------------------------------------------------------
let z={k:1,x:0,y:0};
function applyZ(){vp.setAttribute("transform","translate("+z.x+","+z.y+") scale("+z.k+")")}
function pt(ev){const r=svg.getBoundingClientRect();
  return {x:(ev.clientX-r.left-z.x)/z.k, y:(ev.clientY-r.top-z.y)/z.k}}
svg.addEventListener("wheel",ev=>{ev.preventDefault();
  const s=ev.deltaY<0?1.15:0.87, nk=Math.min(4,Math.max(0.15,z.k*s));
  const r=svg.getBoundingClientRect(),mx=ev.clientX-r.left,my=ev.clientY-r.top;
  z.x=mx-(mx-z.x)*(nk/z.k); z.y=my-(my-z.y)*(nk/z.k); z.k=nk; applyZ();
},{passive:false});
let panning=false,px=0,py=0;
svg.addEventListener("pointerdown",ev=>{if(ev.target===svg||ev.target===vp||ev.target.parentNode===gH){
  panning=true;px=ev.clientX;py=ev.clientY}});
addEventListener("pointermove",ev=>{if(!panning)return;
  z.x+=ev.clientX-px;z.y+=ev.clientY-py;px=ev.clientX;py=ev.clientY;applyZ()});
addEventListener("pointerup",()=>panning=false);
function fit(){
  const vis=nodes.filter(visible); if(!vis.length)return;
  let x0=Infinity,y0=Infinity,x1=-Infinity,y1=-Infinity;
  const grow=(x,y)=>{x0=Math.min(x0,x);y0=Math.min(y0,y);x1=Math.max(x1,x);y1=Math.max(y1,y)};
  vis.forEach(n=>{
    grow(n.x,n.y);
    // A radial label runs OUTWARD from its dot, so the bounding box has to
    // include where the text ends, not where the dot is. Fitting to the
    // dots alone clipped every label on the bottom of the circle.
    if(mode==="radial"&&n._a!==undefined){
      const len=labelExtent(n);
      grow(n.x+Math.cos(n._a)*len, n.y+Math.sin(n._a)*len);
    }
  });
  // ...and the type ring lives outside even those.
  if(mode==="radial"&&radialCentre.band){
    const c=radialCentre, b=c.band+18;
    grow(c.cx-b,c.cy-b); grow(c.cx+b,c.cy+b);
  }
  const pad=mode==="radial"?26:(mode==="timeline"?130:120);
  const bw=Math.max(1,x1-x0+pad*2), bh=Math.max(1,y1-y0+pad*2);
  z.k=Math.min(2.2,Math.max(0.15,Math.min(W/bw,(H-60)/bh)));
  z.x=(W-(x0+x1)*z.k)/2; z.y=60+((H-60)-(y0+y1)*z.k)/2; applyZ();
}
function centerOn(n){z.x=W/2-n.x*z.k;z.y=H/2-n.y*z.k;applyZ()}
$("fit").onclick=fit;
$("viewRadial").onclick=()=>setMode("radial");
$("viewGraph").onclick=()=>setMode("graph");
$("viewTime").onclick=()=>setMode("timeline");

// ---- filters / emphasis ----------------------------------------------------
const hiddenComs=new Set(), hiddenTypes=new Set();
let activeOnly=false, q="", selected=null, hover=null;
function visible(n){
  if(hiddenComs.has(n.community))return false;
  if(hiddenTypes.has(n.type))return false;
  if(activeOnly&&INACTIVE.has(n.status))return false;
  return true;
}
function applyFilters(){
  nodes.forEach(n=>{n._el.style.display=visible(n)?"":"none"});
  links.forEach((l,i)=>eEls[i].style.display=(visible(l.s)&&visible(l.t))?"":"none");
  tlinks.forEach((l,i)=>tEls[i].style.display=(visible(l.s)&&visible(l.t))?"":"none");
  relayout(); applyEmphasis(); renderCounts();
}
function paint(){
  nodes.forEach(n=>{
    const c=colorOf(n.type), com=comOf(n);
    n._disc.setAttribute("fill",c);
    n._ring.setAttribute("stroke",com.color);
    n._ring.setAttribute("stroke-opacity",INACTIVE.has(n.status)?0.3:0.75);
    n._halo.setAttribute("fill",c);
    n._halo.style.display=(n.type==="decision"&&!INACTIVE.has(n.status))?"":"none";
    n._lbl.setAttribute("fill",INACTIVE.has(n.status)?"var(--faint)":"var(--text)");
    n._lbl.setAttribute("stroke","var(--stroke)");
  });
  links.forEach((l,i)=>eEls[i].setAttribute("stroke",colorOf(l.s.type)));
}
function applyEmphasis(){
  const focus=hover||selected;
  const keep=new Set(); if(focus){keep.add(focus);nbrs[focus.id].forEach(m=>keep.add(m.n))}
  nodes.forEach(n=>{
    const match=!q||n.label.toLowerCase().includes(q)||(n.summary||"").toLowerCase().includes(q);
    const dim=(focus&&!keep.has(n))||!match;
    n._el.style.opacity=dim?0.12:1;
  });
  links.forEach((l,i)=>{
    const on=focus&&(l.s===focus||l.t===focus);
    eEls[i].setAttribute("stroke-opacity",focus?(on?0.95:0.04):0.34);
    eEls[i].setAttribute("stroke-width",on?2.4:1.25);
    const lab=eLabels[i];
    if(on&&visible(l.s)&&visible(l.t)){
      lab.style.display="";lab.textContent=l.type.replace(/_/g," ");
      lab.setAttribute("fill",colorOf(l.s.type));
    }else lab.style.display="none";
  });
  tlinks.forEach((l,i)=>tEls[i].setAttribute("stroke-opacity",
    (!focus||l.s===focus||l.t===focus)?0.9:0.1));
  renderHulls();
}
$("activeOnly").onclick=function(){activeOnly=!activeOnly;this.classList.toggle("on",activeOnly);applyFilters()};
$("search").addEventListener("input",function(){q=this.value.trim().toLowerCase();applyEmphasis()});

// ---- panel: headline numbers -----------------------------------------------
const C=META.counts||{};
function kpi(label,value,tone){
  return "<div class='kpi'><span>"+esc(label)+"</span><b"+
    (tone?" class='"+tone+"'":"")+">"+esc(value)+"</b></div>";
}
$("kpis").innerHTML=
  kpi("Nodes",(C.nodes||0).toLocaleString())+
  kpi("Relations",(C.relations||0).toLocaleString())+
  kpi("Decisions",(C.decisions||0).toLocaleString())+
  kpi("Decisions changed",(C.changes||0).toLocaleString())+
  kpi("Open issues",C.open_issues||0,C.open_issues?"bad":"good")+
  // These two are the graph's own integrity, and they were previously
  // reachable only by calling /reasoning and reading a list of anomalies.
  kpi("History gaps",(C.dangling_history||0)+(C.unexplained_supersessions||0),
      ((C.dangling_history||0)+(C.unexplained_supersessions||0))?"bad":"good")+
  kpi("Unconnected nodes",C.unlinked||0,(C.unlinked||0)?"":"good");

// ---- type legend -----------------------------------------------------------
const typeList=$("typeList"), typeRows={};
typesPresent.forEach(t=>{
  const row=document.createElement("label");row.className="row";
  row.innerHTML="<input type='checkbox' checked/><span class='dot'></span>"+
    "<span class='name'>"+esc(t)+"</span><span class='n'></span>";
  const cb=row.querySelector("input");
  cb.onchange=()=>{cb.checked?hiddenTypes.delete(t):hiddenTypes.add(t);
    row.classList.toggle("off",!cb.checked);applyFilters()};
  typeList.appendChild(row); typeRows[t]={row,cb,dot:row.querySelector(".dot")};
});
function paintLegend(){
  typesPresent.forEach(t=>{typeRows[t].dot.style.background=colorOf(t)});
}

// ---- communities panel -----------------------------------------------------
const comList=$("comList"), comRows={};
comIds.forEach(id=>{
  const c=COMS[id];
  const row=document.createElement("label");row.className="row";
  row.innerHTML="<input type='checkbox' checked/><span class='dot' style='background:"+c.color+"'></span>"+
    "<span class='name' title='"+esc(c.name)+"'>"+esc(c.name)+"</span><span class='n'></span>";
  const cb=row.querySelector("input");
  cb.onchange=()=>{cb.checked?hiddenComs.delete(id):hiddenComs.add(id);
    row.classList.toggle("off",!cb.checked);syncSelectAll();applyFilters()};
  comList.appendChild(row); comRows[id]={row,cb};
});
function renderCounts(){
  comIds.forEach(id=>{
    const total=nodes.filter(n=>n.community===id).length;
    const shown=nodes.filter(n=>n.community===id&&visible(n)).length;
    comRows[id].row.querySelector(".n").textContent=shown===total?total:(shown+"/"+total);
  });
  typesPresent.forEach(t=>{
    const total=nodes.filter(n=>n.type===t).length;
    const shown=nodes.filter(n=>n.type===t&&visible(n)).length;
    typeRows[t].row.querySelector(".n").textContent=shown===total?total:(shown+"/"+total);
  });
  const vis=nodes.filter(visible).length;
  $("comTotal").textContent=vis+"/"+nodes.length;
  $("typeTotal").textContent=vis+"/"+nodes.length;
}
function syncSelectAll(){const all=$("selectAll");all.checked=hiddenComs.size===0;
  all.indeterminate=hiddenComs.size>0&&hiddenComs.size<comIds.length}
$("selectAll").onchange=function(){
  if(this.checked)hiddenComs.clear(); else comIds.forEach(id=>hiddenComs.add(id));
  comIds.forEach(id=>{comRows[id].cb.checked=this.checked;comRows[id].row.classList.toggle("off",!this.checked)});
  syncSelectAll();applyFilters();
};

// ---- hotspots and relation flows -------------------------------------------
const hots=META.hotspots||[];
$("hotList").innerHTML=hots.length
  ? hots.map(h=>"<div class='hot' data-label=\""+esc(h.label)+"\">"+
      "<span class='hl'>"+esc(h.label)+"</span>"+
      "<span class='hn'>"+h.depends_on_it+" in</span></div>").join("")
  : "<div class='empty'>Nothing points at anything yet.</div>";
$("hotList").addEventListener("click",ev=>{
  const row=ev.target.closest(".hot"); if(!row)return;
  const n=nodes.find(x=>(x.full_label||x.label)===row.dataset.label);
  if(n){select(n);centerOn(n)}
});
const flows=META.flows||[];
function paintFlows(){
  const max=Math.max.apply(null,flows.map(f=>f.count).concat([1]));
  $("flowList").innerHTML=flows.length
    ? flows.map(f=>"<div class='flow'><span class='fl'>"+esc(f.source)+" &rarr; "+
        esc(f.target)+"</span><span class='ft' style='width:"+
        Math.round(f.count/max*100)+"%;background:"+colorOf(f.source)+
        "'></span><span class='fn'>"+f.count+"</span></div>").join("")
    : "<div class='empty'>No relations extracted yet.</div>";
}

// ---- reading the picture ---------------------------------------------------
function paintStatusLegend(){
  $("statusLegend").innerHTML=[
    ["position","which arc a node sits on IS its type"],
    ["fill","node type, named on the arc and in the legend"],
    ["ring","detected community"],
    ["size","how much weight the session gives it"],
    ["chord","a relation, coloured by where it starts"],
    ["dashed red arc","a decision that replaced another"],
    ["red ring","an open bug"],
    ["dotted amber","two live decisions conflict"],
    ["struck through","superseded or archived"],
  ].map(([k,v])=>"<div class='kv'><b>"+esc(k)+"</b> &middot; "+esc(v)+"</div>").join("");
}

// ---- node detail -----------------------------------------------------------
const detail=$("detail"), detailBody=$("detailBody");
function chainFor(n){
  const hops=(trByNode[n.id]||[]).slice().sort((a,b)=>(a.timestamp||0)-(b.timestamp||0));
  if(!hops.length)return "";
  return "<div class='kv' style='margin-top:9px'>why this is the current answer</div><div class='chain'>"+
    hops.map(t=>"<div class='hop'><span class='old'>"+esc(t.from_label)+"</span>"+
      " <span class='new'>&rarr; "+esc(t.to_label)+"</span>"+
      (t.reason?"<div class='why'>"+esc(t.reason)+"</div>":"")+
      (t.evidence?"<div class='why'>evidence: "+esc(t.evidence)+"</div>":"")+
      (t.trigger?"<div class='why'>trigger: "+esc(t.trigger)+"</div>":"")+"</div>").join("")+"</div>";
}
function select(n){
  selected=n; applyEmphasis();
  if(!n){detail.style.display="none";return}
  const com=comOf(n), tcol=colorOf(n.type);
  const seen={}, neigh=nbrs[n.id].filter(m=>{if(seen[m.n.id+m.rel])return false;
    return (seen[m.n.id+m.rel]=1)}).map(m=>
    "<li data-id='"+esc(m.n.id)+"'><span class='dot' style='width:8px;height:8px;border-radius:50%;"+
    "flex:none;background:"+colorOf(m.n.type)+"'></span><span class='name'>"+esc(m.n.label)+
    "</span><span class='rel'>"+esc(m.rel.replace(/_/g," "))+"</span></li>").join("");
  detailBody.innerHTML="<div class='lbl'>"+esc(n.full_label||n.label)+"</div>"+
    "<span class='pill' style='color:"+tcol+";border-color:"+tcol+"'>"+esc(n.type)+"</span>"+
    "<span class='pill' style='color:var(--muted);border-color:var(--border)'>"+esc(n.status)+"</span>"+
    "<div class='kv'>community <b>"+esc(com.name)+"</b></div>"+
    "<div class='kv'>importance <b>"+n.importance+"</b></div><div class='bar'><i style='width:"+
      Math.round(n.importance*100)+"%;background:"+tcol+"'></i></div>"+
    "<div class='kv'>confidence <b>"+n.confidence+"</b></div><div class='bar'><i style='width:"+
      Math.round(n.confidence*100)+"%;background:"+tcol+"'></i></div>"+
    (n.created_at?"<div class='kv'>first seen <b>"+esc(when(n.created_at))+"</b></div>":"")+
    (n.valid_until?"<div class='kv'>no longer current since <b>"+esc(when(n.valid_until))+"</b></div>":"")+
    (n.summary?"<div class='kv' style='margin-top:7px;color:var(--text)'>"+esc(n.summary)+"</div>":"")+
    chainFor(n)+
    (neigh?"<div class='kv' style='margin-top:9px'>connected to</div><ul class='neighbors'>"+neigh+"</ul>":"");
  detail.style.display="block";
  detailBody.querySelectorAll(".neighbors li").forEach(li=>li.onclick=()=>{
    const m=byId[li.dataset.id];if(m){select(m);centerOn(m)}});
}
svg.addEventListener("click",ev=>{if(ev.target===svg||ev.target===vp)select(null)});

// ---- decision history ------------------------------------------------------
const trList=$("trList");
if(!trans.length){
  trList.innerHTML="<div class='empty'>No decision changes yet.<br>When a decision is superseded, the old&#8594;new story appears here.</div>";
}else{
  [...trans].sort((a,b)=>(b.timestamp||0)-(a.timestamp||0)).forEach(t=>{
    const d=document.createElement("div");d.className="tr-item";
    d.innerHTML="<span class='old'>"+esc(t.from_label)+"</span><span class='arrow'>&#8594;</span>"+
      "<span class='new'>"+esc(t.to_label)+"</span>"+
      (t.reason?"<div class='why'>"+esc(t.reason)+"</div>":"")+
      "<div class='meta'>"+esc(t.trigger||"superseded")+(t.timestamp?" &middot; "+esc(when(t.timestamp)):"")+"</div>";
    d.onclick=()=>{
      document.querySelectorAll(".tr-item.sel").forEach(x=>x.classList.remove("sel"));
      d.classList.add("sel");
      const b=byId[t.to_id]; if(b){select(b);centerOn(b)}
    };
    trList.appendChild(d);
  });
}
document.addEventListener("keydown",e=>{
  if(e.key==="Escape"){document.querySelectorAll(".tr-item.sel").forEach(x=>x.classList.remove("sel"));
    q="";$("search").value="";select(null);applyEmphasis()}
  if(e.key==="f"&&e.target.tagName!=="INPUT")fit();
  if(e.key==="/"&&e.target.tagName!=="INPUT"){e.preventDefault();$("search").focus()}
});

// ---- theme -----------------------------------------------------------------
// Dark is not flipped to light: each is its own validated set of steps for
// its own surface. See the palette note in visualization.py.
$("theme").onclick=function(){
  const light=document.documentElement.getAttribute("data-theme")==="light";
  document.documentElement.setAttribute("data-theme",light?"dark":"light");
  this.textContent=light?"Light":"Dark";
  COLOR=light?COLOR_DARK:COLOR_LIGHT;
  paint();paintLegend();paintFlows();relayout();applyEmphasis();
};

// ---- empty state -----------------------------------------------------------
if(!nodes.length){
  const seen=META.processed_messages||0; let why;
  if(META.load_failed)
    why="<p>The stored graph for this session could not be read (usually another process held the database lock). Nothing was lost; reload once the lock clears.</p>";
  else if(META.data_loss_detected)
    why="<p>Stored memory for this session was displaced during database recovery. See the server log for the quarantined file.</p>";
  else if(META.persistence_broken)
    why="<p>This session is running without durable storage; its memory will not survive a restart. See the server log.</p>";
  else if(!seen)
    why="<p>No messages have been seen for this session.</p>"+
        "<p>Send chat requests through the proxy with <code>session_id</code> set to this id, or post a transcript to <code>POST /api/checkpoint?session_id=...</code> with a <code>messages</code> body.</p>";
  else
    why="<p><b>"+seen+"</b> message"+(seen===1?"":"s")+" processed, nothing extracted.</p>"+
        "<p>The heuristic extractor keys on explicit phrasing: <code>decided to use X</code>, <code>fixed Y</code>, <code>working on Z</code>, and file paths like <code>api/auth.py</code>. For free-form conversation, enable <code>graph_checkpoint.use_llm_extraction</code>; a local model server works with no key.</p>";
  $("emptyBody").innerHTML=why; $("emptyCard").style.display="block";
}

// ---- PNG export ------------------------------------------------------------
$("exportPng").onclick=()=>{
  const light=document.documentElement.getAttribute("data-theme")==="light";
  const clone=svg.cloneNode(true);
  clone.setAttribute("xmlns",NS);
  // Computed theme variables do not survive serialization; bake them.
  clone.querySelectorAll("text").forEach(t=>{
    const f=t.getAttribute("fill")||"";
    if(f.indexOf("var(")===0)t.setAttribute("fill",light?"#12141c":"#e9ecf5");
    const s=t.getAttribute("stroke")||"";
    if(s.indexOf("var(")===0)t.setAttribute("stroke",light?"#ffffff":"#0b0d14");
  });
  clone.querySelectorAll("line,path,circle").forEach(p=>{
    const s=p.getAttribute("stroke")||"";
    if(s.indexOf("var(")===0)p.setAttribute("stroke",light?"#dde1ed":"#242838");
  });
  const bg=document.createElementNS(NS,"rect");
  bg.setAttribute("width","100%");bg.setAttribute("height","100%");
  bg.setAttribute("fill",light?"#f7f8fc":"#0b0d14");
  clone.insertBefore(bg,clone.firstChild);
  const blob=new Blob([new XMLSerializer().serializeToString(clone)],{type:"image/svg+xml"});
  const url=URL.createObjectURL(blob), img=new Image();
  img.onload=()=>{
    const cv=document.createElement("canvas");cv.width=W*2;cv.height=H*2;
    const ctx=cv.getContext("2d");ctx.scale(2,2);ctx.drawImage(img,0,0);
    URL.revokeObjectURL(url);
    const a=document.createElement("a");
    a.download="tokenmizer-"+DATA.session_id.replace(/[^\w.-]/g,"_")+".png";
    a.href=cv.toDataURL("image/png");a.click();
  };
  img.src=url;
};

paint();paintLegend();paintFlows();paintStatusLegend();
layoutRadial();render();applyFilters();fit();
</script></body></html>
"""


def to_share_html(graph: "GraphMemory") -> str:
    """Self-contained dark interactive graph HTML — open in any browser,
    zero network dependencies (works offline / air-gapped demo).

    What it shows that a generic graph view doesn't:
      - nodes filled by detected community, ringed by node type, with a
        side panel that lists the communities (name, count, checkbox)
      - click a node for its detail (type, status, confidence,
        importance, summary, community) and its neighbors
      - decision supersession arcs (old → new, dashed red, arrowhead)
      - a clickable "Decision history" timeline panel (old label struck
        through → new label, with trigger + reason + timestamp); clicking
        an entry selects the new decision and centers the view
      - active decisions get a glow; superseded/archived get a dashed
        ring + strikethrough label; invalidated get a red dashed ring
      - "Active only" toggle, text search, fit-to-view, wheel-zoom/pan,
        drag-to-pin, and one-click PNG export
      - an empty graph explains why, from meta's health fields
    """
    import html as _html
    import json as _json

    # session_id is client-supplied (see security/ownership.py), and node
    # labels/summaries/decision text come straight from conversation
    # content — none of it is safe to drop into this template unescaped.
    # This is a genuine injection surface, not a theoretical one: the
    # docstring above literally says "share" — the intended use is
    # downloading this file and handing it to someone else, so a payload
    # here isn't just self-XSS, it runs in whoever opens the shared file.
    #
    # Two different escaping rules for two different sink contexts, not
    # one shared substitution:
    #  - __SESSION__ appears in plain HTML text (title, header) — needs
    #    HTML escaping. It used to ALSO appear inside a JS string literal
    #    (the PNG filename) with the same unescaped value; that call site
    #    now reads DATA.session_id instead of a template-substituted raw
    #    string, so there is exactly one sink left needing exactly one
    #    escaping rule.
    #  - __DATA__/__COLORS__ go inside a <script> block as JSON. json.dumps
    #    already makes the result valid, safely-quoted JS — but it does
    #    NOT escape "</", so a label containing the literal text
    #    "</script>" would still terminate the block early and let
    #    whatever followed run as HTML. Escaping "<" to its unicode
    #    escape blocks that (and "<!--") without changing the decoded
    #    value on the JS side.
    def _script_json(value) -> str:
        return _json.dumps(value).replace("<", "\\u003c")

    vis = graph.to_vis_json()
    nodes = vis.get("nodes", [])
    edges = vis.get("edges", [])
    transitions = vis.get("transitions", [])
    decisions = sum(1 for n in nodes if n.get("type") == "decision")
    html = (_SHARE_HTML_TEMPLATE
            .replace("__SESSION__", _html.escape(graph.session_id, quote=True))
            .replace("__NODES__", str(len(nodes)))
            .replace("__EDGES__", str(len(edges)))
            .replace("__DECISIONS__", str(decisions))
            .replace("__TRANSITIONS__", str(len(transitions)))
            .replace("__DATA__", _script_json({
                "nodes": nodes, "edges": edges, "transitions": transitions,
                "session_id": graph.session_id, "meta": vis.get("meta", {}),
            }))
            .replace("__COLORS__", _script_json(_TYPE_COLOR))
            .replace("__COLORS_LIGHT__", _script_json(_TYPE_COLOR_LIGHT)))
    return html
