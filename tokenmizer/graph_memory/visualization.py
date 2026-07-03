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


_TYPE_COLOR = {
    "goal":        "#e879f9",
    "task":        "#4ade80",
    "decision":    "#a78bfa",
    "file":        "#60a5fa",
    "error":       "#f87171",
    "dependency":  "#fbbf24",
    "environment": "#5ee7c8",
    "endpoint":    "#38bdf8",
    "schema":      "#fb923c",
}

_TYPE_SIZE = {
    "goal": 22, "decision": 18, "task": 14,
    "error": 14, "endpoint": 12, "schema": 12,
    "file": 10, "dependency": 9, "environment": 9,
}

_STATUS_OPACITY = {
    "completed": 1.0, "in_progress": 0.9, "pending": 0.7,
    "failed": 0.6, "superseded": 0.35, "archived": 0.25,
    "modified": 0.5, "invalidated": 0.2,
}

_EDGE_COLOR = {
    "related_to":   "#8b8fa8",
    "implements":   "#60a5fa",
    "part_of":      "#a78bfa",
    "depends_on":   "#fbbf24",
    "supersedes":   "#f87171",
    "references":   "#5ee7c8",
    "derived_from": "#fb923c",
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
}

_EDGE_LABEL = {
    "related_to": "related", "implements": "implements",
    "part_of": "part of", "depends_on": "depends on",
    "supersedes": "supersedes", "references": "ref",
    "derived_from": "derived",
}


def to_vis_json(graph: "GraphMemory") -> dict:
    """
    Export graph as D3-compatible JSON: {nodes, edges, transitions, meta}.
    Node colors and sizes encode type + status.
    Only exports active (non-evicted, non-archived) nodes.
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

    return {
        "session_id":  graph.session_id,
        "nodes":       vis_nodes,
        "edges":       vis_edges,
        "transitions": vis_transitions,
        "meta": {
            "node_count":       len(vis_nodes),
            "edge_count":       len(vis_edges),
            "transition_count": len(vis_transitions),
            "by_type": {
                t: sum(1 for n in vis_nodes if n["type"] == t)
                for t in _TYPE_COLOR
                if any(n["type"] == t for n in vis_nodes)
            },
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
        "🟣 GOAL  🟡 DECISION  🟢 TASK",
        "🔵 FILE  🔴 ERROR  🩵 ENDPOINT",
    ]
    canvas_nodes.insert(0, {
        "id": "legend", "type": "text",
        "x": -300, "y": 0, "width": 240, "height": 200,
        "color": "6", "text": "\n".join(legend),
    })

    return {"nodes": canvas_nodes, "edges": canvas_edges}


# ── Shareable standalone HTML (the "look at my session's brain" artifact) ────

_SHARE_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>TokenMizer — __SESSION__</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<style>
 body{margin:0;background:#0d1117;color:#c9d1d9;font:14px/1.4 'Segoe UI',system-ui,sans-serif;overflow:hidden}
 #hdr{position:fixed;top:0;left:0;right:0;padding:14px 22px;display:flex;gap:18px;align-items:baseline;
      background:linear-gradient(#0d1117ee,#0d111700);z-index:2;pointer-events:none}
 #hdr b{font-size:18px;color:#e6edf3}
 #hdr .stat{color:#8b949e}#hdr .stat i{color:#5ee7c8;font-style:normal;font-weight:600}
 #legend{position:fixed;bottom:44px;left:22px;z-index:2;display:flex;flex-wrap:wrap;gap:10px;max-width:70vw}
 #legend span{display:flex;align-items:center;gap:5px;color:#8b949e;font-size:12px}
 #legend i{width:10px;height:10px;border-radius:50%;display:inline-block}
 #ftr{position:fixed;bottom:12px;left:22px;color:#484f58;font-size:12px;z-index:2}
 #ftr a{color:#7c6af7;text-decoration:none}
 .node text{fill:#c9d1d9;font-size:11px;pointer-events:none;text-shadow:0 1px 3px #000}
 svg{cursor:grab}svg:active{cursor:grabbing}
</style></head><body>
<div id="hdr"><b>🧠 __SESSION__</b>
 <span class="stat"><i>__NODES__</i> nodes</span>
 <span class="stat"><i>__EDGES__</i> edges</span>
 <span class="stat"><i>__DECISIONS__</i> decisions tracked</span></div>
<div id="legend"></div>
<div id="ftr">session memory graph · <a href="https://github.com/Shweta-Mishra-ai/tokenmizer">TokenMizer</a> · pip install tokenmizer</div>
<script>
const data = __DATA__;
const COLOR = __COLORS__;
const W=innerWidth,H=innerHeight;
const svg=d3.select("body").append("svg").attr("width",W).attr("height",H);
const defs=svg.append("defs");
const glow=defs.append("filter").attr("id","glow");
glow.append("feGaussianBlur").attr("stdDeviation","4").attr("result","b");
const m=glow.append("feMerge");m.append("feMergeNode").attr("in","b");m.append("feMergeNode").attr("in","SourceGraphic");
const g=svg.append("g");
svg.call(d3.zoom().scaleExtent([0.2,4]).on("zoom",e=>g.attr("transform",e.transform)));
const link=g.selectAll("line").data(data.edges).join("line")
 .attr("stroke",d=>d.color||"#30363d").attr("stroke-opacity",0.45).attr("stroke-width",1.2);
const node=g.selectAll("g.node").data(data.nodes).join("g").attr("class","node")
 .call(d3.drag().on("start",(e,d)=>{if(!e.active)sim.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y})
 .on("drag",(e,d)=>{d.fx=e.x;d.fy=e.y})
 .on("end",(e,d)=>{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null}));
node.append("circle").attr("r",d=>d.size||10).attr("fill",d=>COLOR[d.type]||"#8b949e")
 .attr("fill-opacity",d=>d.opacity??0.9).attr("filter","url(#glow)")
 .append("title").text(d=>d.type+": "+d.label+(d.summary?" — "+d.summary:""));
node.append("text").attr("dx",d=>(d.size||10)+4).attr("dy",4)
 .text(d=>d.label.length>34?d.label.slice(0,32)+"…":d.label);
const sim=d3.forceSimulation(data.nodes)
 .force("link",d3.forceLink(data.edges).id(d=>d.id).distance(90).strength(0.4))
 .force("charge",d3.forceManyBody().strength(-260))
 .force("center",d3.forceCenter(W/2,H/2))
 .force("collide",d3.forceCollide().radius(d=>(d.size||10)+14))
 .on("tick",()=>{link.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
 .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
 node.attr("transform",d=>`translate(${d.x},${d.y})`)});
const types=[...new Set(data.nodes.map(d=>d.type))];
d3.select("#legend").selectAll("span").data(types).join("span")
 .html(t=>`<i style="background:${COLOR[t]||"#8b949e"}"></i>${t}`);
</script></body></html>
"""


def to_share_html(graph: "GraphMemory") -> str:
    """Self-contained dark interactive force-graph HTML — open in any
    browser, drag/zoom, screenshot, share. The viral artifact."""
    import json as _json

    vis = graph.to_vis_json()
    nodes = vis.get("nodes", [])
    edges = vis.get("edges", [])
    # d3.forceLink needs source/target keys
    for e in edges:
        e.setdefault("source", e.get("from") or e.get("source_id"))
        e.setdefault("target", e.get("to") or e.get("target_id"))
    decisions = sum(1 for n in nodes if n.get("type") == "decision")
    html = (_SHARE_HTML_TEMPLATE
            .replace("__SESSION__", graph.session_id)
            .replace("__NODES__", str(len(nodes)))
            .replace("__EDGES__", str(len(edges)))
            .replace("__DECISIONS__", str(decisions))
            .replace("__DATA__", _json.dumps({"nodes": nodes, "edges": edges}))
            .replace("__COLORS__", _json.dumps(_TYPE_COLOR)))
    return html
