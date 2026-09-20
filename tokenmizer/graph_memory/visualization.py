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


# One entry per NodeType, enforced by tests/unit/test_visualization.py:
# to_vis_json's meta.by_type iterates this map, so a type missing here
# was not only drawn grey but dropped from the counts as well.
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
    "concept":     "#c084fc",
    "api":         "#22d3ee",
    "project":     "#f472b6",
    "agent":       "#a3e635",
    "test":        "#facc15",
}

_TYPE_SIZE = {
    "goal": 22, "decision": 18, "project": 16, "task": 14,
    "error": 14, "endpoint": 12, "schema": 12, "api": 12,
    "concept": 11, "agent": 11, "test": 10,
    "file": 10, "dependency": 9, "environment": 9,
}

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
            "communities":        communities,
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
 :root{--bg:#0f1117;--surface:#1a1d27;--surface2:#151822;--border:#2a2d3e;--accent:#7c6af7;
       --accent2:#5ee7c8;--text:#e8eaf6;--muted:#8b8fa8;--faint:#5b6078;--red:#f87171;
       --shadow:0 8px 28px #0009;--side:350px;--stroke:#0f1117}
 html[data-theme="light"]{--bg:#f7f8fc;--surface:#ffffff;--surface2:#f0f2f8;--border:#dfe3ef;
       --text:#1a1d27;--muted:#646a85;--faint:#9aa0b8;--shadow:0 8px 28px #1a1d2718;--stroke:#ffffff}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--text);overflow:hidden;
      font:14px/1.45 'Inter',system-ui,-apple-system,sans-serif;
      transition:background .2s,color .2s}
 #hdr{position:fixed;top:0;left:0;right:var(--side);padding:12px 20px;display:flex;gap:10px;
      align-items:center;flex-wrap:wrap;z-index:3;pointer-events:none}
 #hdr>*{pointer-events:auto}
 #hdr b{font-size:17px;font-weight:650;margin-right:2px;letter-spacing:-.01em}
 .stat{color:var(--muted);font-size:12px;white-space:nowrap}
 .stat i{color:var(--accent2);font-style:normal;font-weight:650}
 #search{background:var(--surface);border:1px solid var(--border);color:var(--text);
      border-radius:8px;padding:6px 10px;font:inherit;font-size:12px;width:170px;outline:none;margin-left:auto}
 #search:focus{border-color:var(--accent)}
 .btn{background:var(--surface);border:1px solid var(--border);color:var(--text);border-radius:8px;
      padding:6px 11px;font-size:12px;cursor:pointer;user-select:none;white-space:nowrap;
      transition:border-color .15s,color .15s}
 .btn:hover{border-color:var(--accent)}
 .btn.on{border-color:var(--accent2);color:var(--accent2)}
 .seg{display:flex;border:1px solid var(--border);border-radius:8px;overflow:hidden;background:var(--surface)}
 .seg .btn{border:0;border-radius:0;padding:6px 12px}
 .seg .btn.on{background:var(--accent);color:#fff}
 #side{position:fixed;top:0;right:0;bottom:0;width:var(--side);background:var(--surface);
      border-left:1px solid var(--border);z-index:4;overflow-y:auto;padding:14px 18px 28px}
 #side h2{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);
      margin:20px 0 8px;display:flex;justify-content:space-between;align-items:center}
 #side h2:first-child{margin-top:2px}
 #side h2 span{text-transform:none;letter-spacing:0;font-size:11px;color:var(--faint)}
 .row{display:flex;align-items:center;gap:8px;padding:5px 6px;border-radius:6px;cursor:pointer;
      font-size:13px;user-select:none}
 .row:hover{background:var(--surface2)}
 .row .dot{width:10px;height:10px;border-radius:50%;flex:none}
 .row .sq{width:10px;height:10px;border-radius:3px;flex:none;border:2px solid}
 .row .name{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
 .row .n{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
 .row.off .name,.row.off .dot,.row.off .sq,.row.off .n{opacity:.32}
 .row input{accent-color:var(--accent);margin:0;flex:none}
 #detail{display:none}
 #detail .lbl{font-size:14px;font-weight:600;margin-bottom:8px;word-break:break-word;line-height:1.35}
 .pill{display:inline-block;font-size:11px;padding:1px 8px;border-radius:10px;border:1px solid;margin:0 6px 6px 0}
 .kv{color:var(--muted);font-size:12px;margin:3px 0}.kv b{color:var(--text);font-weight:500}
 .bar{height:4px;border-radius:2px;background:var(--border);margin:2px 0 7px;overflow:hidden}
 .bar i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2))}
 .neighbors{list-style:none;margin:6px 0 0;padding:0}
 .neighbors li{padding:4px 6px;border-radius:6px;cursor:pointer;font-size:12px;display:flex;gap:7px;align-items:center}
 .neighbors li:hover{background:var(--surface2)}
 .neighbors li .name{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1}
 .neighbors li .rel{color:var(--faint);font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;flex:none}
 .chain{border-left:2px solid var(--red);margin:8px 0 0 4px;padding:2px 0 2px 10px}
 .chain .hop{font-size:12px;margin-bottom:7px}
 .chain .hop .old{color:var(--muted);text-decoration:line-through}
 .chain .hop .new{font-weight:600}
 .chain .hop .why{color:var(--muted);font-size:11.5px;margin-top:2px}
 .tr-item{border:1px solid var(--border);border-radius:8px;padding:10px 12px;margin-bottom:10px;
      cursor:pointer;transition:border-color .15s}
 .tr-item:hover,.tr-item.sel{border-color:var(--red)}
 .tr-item .old{color:var(--muted);text-decoration:line-through;font-size:12px}
 .tr-item .arrow{color:var(--red);margin:0 4px}
 .tr-item .new{font-size:13px;font-weight:600}
 .tr-item .why{color:var(--muted);font-size:12px;margin-top:5px}
 .tr-item .meta{color:var(--faint);font-size:11px;margin-top:4px}
 .empty{color:var(--muted);font-size:13px;border:1px dashed var(--border);border-radius:8px;
      padding:16px;text-align:center;line-height:1.5}
 #emptyCard{position:fixed;top:50%;left:calc((100vw - var(--side)) / 2);transform:translate(-50%,-50%);
      width:min(560px,80vw);background:var(--surface);border:1px solid var(--border);border-radius:14px;
      padding:24px 26px;z-index:3;display:none;box-shadow:var(--shadow)}
 #emptyCard h3{margin:0 0 10px;font-size:17px}
 #emptyCard p{color:var(--muted);font-size:13px;margin:9px 0;line-height:1.6}
 #emptyCard code{background:var(--surface2);padding:1px 6px;border-radius:4px;font-size:12px;color:var(--text)}
 #tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--border);
      border-radius:9px;padding:9px 12px;font-size:12px;max-width:320px;z-index:9;display:none;
      box-shadow:var(--shadow);line-height:1.45}
 #tip b{color:var(--text)}#tip .st{color:var(--accent2)}
 #ftr{position:fixed;bottom:12px;left:20px;color:var(--faint);font-size:12px;z-index:3}
 #ftr a{color:var(--accent);text-decoration:none}
 svg{cursor:grab;display:block}svg:active{cursor:grabbing}
 .lane{fill:var(--faint);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase}
 .axis{stroke:var(--border);stroke-width:1}
 .axtx{fill:var(--faint);font-size:10.5px}
</style></head><body>
<div id="hdr"><b>__SESSION__</b>
 <span class="stat"><i>__NODES__</i> nodes</span>
 <span class="stat"><i>__EDGES__</i> edges</span>
 <span class="stat"><i>__DECISIONS__</i> decisions</span>
 <span class="stat"><i>__TRANSITIONS__</i> changed</span>
 <input id="search" type="search" placeholder="Search nodes"/>
 <span class="seg"><span class="btn on" id="viewGraph">Graph</span><span class="btn" id="viewTime">Timeline</span></span>
 <span class="btn" id="activeOnly">Active only</span>
 <span class="btn" id="fit">Fit</span>
 <span class="btn" id="theme">Light</span>
 <span class="btn" id="exportPng">PNG</span>
</div>
<div id="emptyCard"><h3>No nodes yet</h3><div id="emptyBody"></div></div>
<div id="side">
 <h2>Node types <span id="typeTotal"></span></h2>
 <div id="typeList"></div>
 <h2>Communities <span id="comTotal"></span></h2>
 <label class="row" id="selectAllRow"><input type="checkbox" id="selectAll" checked/>
  <span class="name">Select all</span></label>
 <div id="comList"></div>
 <div id="detail"><h2>Node</h2><div id="detailBody"></div></div>
 <h2>Decision history</h2>
 <div id="trList"></div>
 <h2>Legend</h2>
 <div id="statusLegend"></div>
</div>
<div id="ftr">session memory graph &middot; <a href="https://github.com/Shweta-Mishra-ai/tokenmizer">TokenMizer</a> &middot; pip install tokenmizer</div>
<div id="tip"></div>
<script>
"use strict";
const DATA=__DATA__, COLOR=__COLORS__;
const META=DATA.meta||{};
const INACTIVE=new Set(["superseded","archived","invalidated","modified"]);
const SIDE=350;
let W=Math.max(640,(innerWidth||1280)-SIDE), H=Math.max(480,innerHeight||800);
const NS="http://www.w3.org/2000/svg";
function el(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function $(id){return document.getElementById(id)}
function when(ts){return ts?new Date(ts*1000).toLocaleString():""}

const svg=el("svg",{width:W,height:H,"font-family":"Inter,system-ui,sans-serif"},document.body);
addEventListener("resize",()=>{W=Math.max(640,innerWidth-SIDE);H=Math.max(480,innerHeight);
  svg.setAttribute("width",W);svg.setAttribute("height",H)});
const defs=el("defs",{},svg);
for(const [id,col] of [["arr","#f87171"],["arrDim","#8b8fa8"]]){
  const mk=el("marker",{id:id,viewBox:"0 -5 10 10",refX:18,refY:0,markerWidth:7,markerHeight:7,orient:"auto"},defs);
  el("path",{d:"M0,-5L10,0L0,5",fill:col},mk);
}
const vp=el("g",{},svg);
const gH=el("g",{},vp), gAx=el("g",{},vp), gE=el("g",{},vp),
      gT=el("g",{},vp), gL=el("g",{},vp), gN=el("g",{},vp);

// ---- data ------------------------------------------------------------------
const nodes=DATA.nodes, edges=DATA.edges, trans=DATA.transitions||[];
const byId={}; nodes.forEach(n=>byId[n.id]=n);
const COMS={}; (META.communities||[]).forEach(c=>COMS[c.id]=c);
function comOf(n){return COMS[n.community]||{id:-1,name:"Unclustered",color:"#8b8fa8"}}
const links=edges.map(e=>({s:byId[e.source],t:byId[e.target],color:e.color,type:e.type}))
                 .filter(l=>l.s&&l.t);
const tlinks=trans.map(t=>({s:byId[t.from_id],t:byId[t.to_id],tr:t}))
                  .filter(l=>l.s&&l.t);
const nbrs={}; nodes.forEach(n=>nbrs[n.id]=[]);
links.forEach(l=>{nbrs[l.s.id].push({n:l.t,rel:l.type,dir:"out"});
                  nbrs[l.t.id].push({n:l.s,rel:l.type,dir:"in"})});
const trByNode={}; trans.forEach(t=>{(trByNode[t.from_id]=trByNode[t.from_id]||[]).push(t);
                                     (trByNode[t.to_id]=trByNode[t.to_id]||[]).push(t)});

// Lane order for timeline mode: the order a person reads a session in.
const LANES=["goal","decision","task","error","endpoint","schema","file","dependency",
             "environment","test","concept","api","project","agent"];
const typesPresent=LANES.filter(t=>nodes.some(n=>n.type===t))
      .concat([...new Set(nodes.map(n=>n.type))].filter(t=>!LANES.includes(t)));

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
});

// ---- force simulation ------------------------------------------------------
let alpha=1, mode="graph";
const centroids={};
function step(){
  if(mode==="timeline")return;
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
// A session read as a story: time on x, node type in its own lane on y.
// Nothing else shows WHEN a decision was taken relative to the error that
// caused it, which is the question the graph exists to answer.
function layoutTimeline(){
  const vis=nodes.filter(visible);
  const times=vis.map(n=>n.created_at||n.valid_from||0).filter(t=>t>0);
  let t0=times.length?Math.min.apply(null,times):0;
  let t1=times.length?Math.max.apply(null,times):0;
  // A whole transcript checkpointed in one call gives every node the same
  // created_at, and a time axis over a span of zero is a lie drawn to
  // four decimal places. The order nodes entered the graph IS the order
  // the transcript stated them, so fall back to that and say so on the
  // axis rather than inventing dates. The bar is a minute, not a second:
  // below that every tick renders the same clock time and the axis looks
  // broken rather than degenerate.
  const byTime=(t1-t0)>=60;
  if(!byTime){t0=0;t1=Math.max(1,vis.length-1)}
  const ordinal={}; nodes.forEach((n,i)=>ordinal[n.id]=i);
  const posOf=n=>byTime?(n.created_at||n.valid_from||t0):ordinal[n.id];
  const lanes=typesPresent.filter(t=>vis.some(n=>n.type===t));
  // Lanes need to be tall enough for the rows the spreading below adds,
  // which is driven by how many nodes crowd the busiest lane.
  const busiest=Math.max.apply(null,
    lanes.map(t=>vis.filter(n=>n.type===t).length).concat([1]));
  const laneH=Math.max(64,Math.min(150,(H-170)/Math.max(lanes.length,1),
                                   28+Math.min(busiest,6)*17));
  // Room on the right for a label to run without sliding under the panel.
  const x0=150, x1=Math.max(x0+320,W-230);
  // Nodes from one turn share a position, and a label is ~34 characters
  // wide, so placing them on one row stacks the text into an unreadable
  // smear. Each lane keeps a set of rows; a node takes the first row
  // whose last label ended before this one starts.
  const LABEL_W=210, ROW_H=17;
  const rows={};
  vis.slice().sort((a,b)=>posOf(a)-posOf(b)).forEach(n=>{
    const li=lanes.indexOf(n.type);
    const frac=t1>t0?(posOf(n)-t0)/(t1-t0):0.5;
    const x=x0+frac*(x1-x0);
    const lane=rows[n.type]||(rows[n.type]=[]);
    let row=lane.findIndex(lastRight=>x>=lastRight);
    if(row===-1){row=lane.length;lane.push(0)}
    lane[row]=x+LABEL_W;
    n.fx=x; n.fy=100+li*laneH+row*ROW_H-(laneH/2-14);
    n.x=n.fx;n.y=n.fy;
  });
  gAx.innerHTML="";
  lanes.forEach((t,i)=>{
    const y=100+i*laneH;
    el("line",{x1:x0-38,y1:y,x2:x1+40,y2:y,class:"axis","stroke-opacity":.45},gAx);
    const lab=el("text",{x:x0-46,y:y+4,"text-anchor":"end",class:"lane",fill:COLOR[t]||"#8b8fa8"},gAx);
    lab.textContent=t;
  });
  if(t1>t0){
    for(let i=0;i<=4;i++){
      const x=x0+(x1-x0)*i/4, v=t0+(t1-t0)*i/4;
      el("line",{x1:x,y1:72,x2:x,y2:100+lanes.length*laneH-laneH+40,class:"axis","stroke-opacity":.18},gAx);
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
function setMode(m){
  mode=m;
  $("viewGraph").classList.toggle("on",m==="graph");
  $("viewTime").classList.toggle("on",m==="timeline");
  gH.style.display=m==="graph"?"":"none";
  if(m==="timeline"){layoutTimeline();render();fit()}
  else{gAx.innerHTML="";nodes.forEach(n=>{n.fx=null;n.fy=null});alpha=0.55;requestAnimationFrame(tick);
       setTimeout(fit,260)}
}

// ---- render ----------------------------------------------------------------
const eEls=links.map(l=>el("line",{stroke:l.color||"#2a2d3e","stroke-opacity":0.32,"stroke-width":1.2},gE));
const eLabels=links.map(()=>{const t=el("text",{class:"axtx","text-anchor":"middle"},gL);t.style.display="none";return t});
const tEls=tlinks.map(l=>{
  const p=el("path",{fill:"none",stroke:"#f87171","stroke-width":1.6,
    "stroke-dasharray":"6 4","stroke-opacity":0.85,"marker-end":"url(#arr)"},gT);
  p.dataset.trid=l.tr.id; return p;
});
const hullEls={};
comIds.forEach(id=>{hullEls[id]=el("path",{fill:COMS[id].color,"fill-opacity":0.07,
  stroke:COMS[id].color,"stroke-opacity":0.22,"stroke-width":1.5,"stroke-linejoin":"round"},gH)});

const nEls=nodes.map(n=>{
  const g=el("g",{cursor:"pointer"},gN);
  const r=n.size||10, inactive=INACTIVE.has(n.status), com=comOf(n);
  if(n.type==="decision"&&!inactive)
    el("circle",{r:r+6,fill:com.color,"fill-opacity":0.18},g);
  el("circle",{r:r,fill:com.color,"fill-opacity":inactive?0.22:(n.opacity??0.95)},g);
  el("circle",{r:r+1.5,fill:"none",stroke:COLOR[n.type]||"#8b8fa8","stroke-width":1.5,
    "stroke-opacity":inactive?0.4:0.9},g);
  if(inactive)
    el("circle",{r:r+4,fill:"none",stroke:"#8b8fa8","stroke-dasharray":"3 3","stroke-opacity":0.55,"stroke-width":1},g);
  if(n.status==="invalidated")
    el("circle",{r:r+4,fill:"none",stroke:"#f87171","stroke-dasharray":"2 2","stroke-width":1.4},g);
  if(n.status==="failed")
    el("circle",{r:r+4,fill:"none",stroke:"#f87171","stroke-opacity":.7,"stroke-width":1.2},g);
  if(n.status==="contested")
    el("circle",{r:r+4,fill:"none",stroke:"#fb923c","stroke-dasharray":"1 3","stroke-width":1.6},g);
  const lbl=el("text",{dx:r+6,dy:4,"font-size":11,"paint-order":"stroke",
    "stroke-width":3.5,"stroke-linejoin":"round"},g);
  lbl.textContent=n.label.length>34?n.label.slice(0,32)+"…":n.label;
  if(inactive)lbl.setAttribute("text-decoration","line-through");
  n._lbl=lbl;
  g._n=n; n._el=g;
  let drag=false, moved=false;
  g.addEventListener("pointerdown",ev=>{drag=true;moved=false;g.setPointerCapture(ev.pointerId);ev.stopPropagation()});
  g.addEventListener("pointermove",ev=>{if(!drag)return;
    const m=pt(ev);n.fx=m.x;n.fy=m.y;n.x=m.x;n.y=m.y;moved=true;
    if(mode==="graph")reheat(0.3);else render()});
  g.addEventListener("pointerup",()=>{drag=false;if(!moved)select(n)});
  g.addEventListener("dblclick",ev=>{ev.stopPropagation();
    if(mode==="graph"){n.fx=null;n.fy=null;reheat(0.3)}});
  g.addEventListener("pointerenter",ev=>{
    hover=n;applyEmphasis();
    tip.style.display="block";moveTip(ev);
    const changed=(trByNode[n.id]||[]).length;
    tip.innerHTML="<b>"+esc(n.full_label||n.label)+"</b><br><span class='st'>"+esc(n.type)+
      " &middot; "+esc(n.status)+"</span>"+(n.summary?"<br>"+esc(n.summary):"")+
      "<br><span style='color:var(--faint)'>"+esc(com.name)+" &middot; importance "+n.importance+
      " &middot; confidence "+n.confidence+(changed?" &middot; "+changed+" change"+(changed>1?"s":""):"")+"</span>";
  });
  g.addEventListener("pointermove",moveTip);
  g.addEventListener("pointerleave",()=>{hover=null;applyEmphasis();tip.style.display="none"});
  return g;
});
const tip=$("tip");
function moveTip(ev){
  const w=330, x=Math.min(ev.clientX+14, innerWidth-SIDE-w);
  tip.style.left=Math.max(8,x)+"px";
  tip.style.top=Math.min(ev.clientY+10, innerHeight-140)+"px";
}

// Convex hull (monotone chain) + padding, so a community reads as one
// region instead of a colour the eye has to collect dot by dot.
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
  if(mode!=="graph")return;
  comIds.forEach(id=>{
    const c=COMS[id], path=hullEls[id];
    const pts=nodes.filter(n=>n.community===id&&visible(n)&&nbrs[n.id].length)
                   .map(n=>[n.x,n.y]);
    const d=(c.name==="Unclustered")?null:hull(pts,34);
    if(d){path.setAttribute("d",d);path.style.display=""}else path.style.display="none";
  });
}
function render(){
  links.forEach((l,i)=>{const e=eEls[i];
    e.setAttribute("x1",l.s.x);e.setAttribute("y1",l.s.y);
    e.setAttribute("x2",l.t.x);e.setAttribute("y2",l.t.y);
    const lab=eLabels[i];
    if(lab.style.display!=="none"){
      lab.setAttribute("x",(l.s.x+l.t.x)/2);lab.setAttribute("y",(l.s.y+l.t.y)/2-4);
    }});
  tlinks.forEach((l,i)=>{
    const mx=(l.s.x+l.t.x)/2, my=(l.s.y+l.t.y)/2;
    const dx=l.t.x-l.s.x, dy=l.t.y-l.s.y, d=Math.sqrt(dx*dx+dy*dy)||1;
    tEls[i].setAttribute("d","M"+l.s.x+","+l.s.y+" Q"+(mx-dy/d*40)+","+(my+dx/d*40)+" "+l.t.x+","+l.t.y)});
  nodes.forEach(n=>n._el.setAttribute("transform","translate("+n.x+","+n.y+")"));
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
  vis.forEach(n=>{x0=Math.min(x0,n.x);y0=Math.min(y0,n.y);x1=Math.max(x1,n.x);y1=Math.max(y1,n.y)});
  const padL=mode==="timeline"?150:130, pad=120;
  const bw=Math.max(1,x1-x0+padL+pad), bh=Math.max(1,y1-y0+pad*2);
  z.k=Math.min(2.2,Math.max(0.15,Math.min(W/bw,(H-70)/bh)));
  z.x=(W-(x0+x1)*z.k)/2+(mode==="timeline"?30:0); z.y=70+((H-70)-(y0+y1)*z.k)/2; applyZ();
}
function centerOn(n){z.x=W/2-n.x*z.k;z.y=H/2-n.y*z.k;applyZ()}
$("fit").onclick=fit;
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
  if(mode==="timeline")layoutTimeline();
  applyEmphasis(); renderCounts(); render();
}
function applyEmphasis(){
  const focus=hover||selected;
  const keep=new Set(); if(focus){keep.add(focus);nbrs[focus.id].forEach(m=>keep.add(m.n))}
  nodes.forEach(n=>{
    const match=!q||n.label.toLowerCase().includes(q)||(n.summary||"").toLowerCase().includes(q);
    const dim=(focus&&!keep.has(n))||!match;
    n._el.style.opacity=dim?0.1:1;
    n._lbl.setAttribute("fill",INACTIVE.has(n.status)?"var(--faint)":"var(--text)");
    n._lbl.setAttribute("stroke","var(--stroke)");
  });
  links.forEach((l,i)=>{
    const on=focus&&(l.s===focus||l.t===focus);
    eEls[i].setAttribute("stroke-opacity",focus?(on?0.95:0.05):0.32);
    eEls[i].setAttribute("stroke-width",on?2.2:1.2);
    // Relation names only where they can be read: on the focused node's
    // own edges. Drawn for every edge they are noise at any zoom.
    const lab=eLabels[i];
    if(on&&visible(l.s)&&visible(l.t)){
      lab.style.display="";lab.textContent=l.type.replace(/_/g," ");
      lab.setAttribute("fill",l.color||"var(--muted)");
    }else lab.style.display="none";
  });
  tlinks.forEach((l,i)=>{
    const on=!focus||l.s===focus||l.t===focus;
    tEls[i].setAttribute("stroke-opacity",on?0.85:0.12);
  });
  if(mode==="graph")renderHulls();
}
$("activeOnly").onclick=function(){activeOnly=!activeOnly;this.classList.toggle("on",activeOnly);applyFilters()};
$("search").addEventListener("input",function(){q=this.value.trim().toLowerCase();applyEmphasis()});

// ---- type legend -----------------------------------------------------------
const typeList=$("typeList"), typeRows={};
typesPresent.forEach(t=>{
  const row=document.createElement("label");row.className="row";
  row.innerHTML="<input type='checkbox' checked/><span class='sq' style='border-color:"+
    (COLOR[t]||"#8b8fa8")+";background:"+(COLOR[t]||"#8b8fa8")+"33'></span>"+
    "<span class='name'>"+esc(t)+"</span><span class='n'></span>";
  const cb=row.querySelector("input");
  cb.onchange=()=>{cb.checked?hiddenTypes.delete(t):hiddenTypes.add(t);
    row.classList.toggle("off",!cb.checked);applyFilters()};
  typeList.appendChild(row); typeRows[t]={row,cb};
});

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

// ---- status legend ---------------------------------------------------------
$("statusLegend").innerHTML=[
  ["completed / active","solid fill, full weight"],
  ["in_progress / pending","solid fill, lighter"],
  ["failed","red ring — an open bug"],
  ["contested","dotted orange ring — two live decisions conflict"],
  ["superseded / archived","dashed grey ring, struck-through label"],
  ["invalidated","red dashed ring — do not revisit"],
  ["fill colour","detected community"],
  ["ring colour","node type"],
].map(([k,v])=>"<div class='kv'><b>"+esc(k)+"</b> &middot; "+esc(v)+"</div>").join("");

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
  const com=comOf(n), tcol=COLOR[n.type]||"#8b8fa8";
  const seen={}, neigh=nbrs[n.id].filter(m=>{if(seen[m.n.id+m.rel])return false;
    return (seen[m.n.id+m.rel]=1)}).map(m=>
    "<li data-id='"+esc(m.n.id)+"'><span class='dot' style='width:8px;height:8px;border-radius:50%;"+
    "flex:none;background:"+comOf(m.n).color+"'></span><span class='name'>"+esc(m.n.label)+
    "</span><span class='rel'>"+esc(m.rel.replace(/_/g," "))+"</span></li>").join("");
  detailBody.innerHTML="<div class='lbl'>"+esc(n.full_label||n.label)+"</div>"+
    "<span class='pill' style='color:"+tcol+";border-color:"+tcol+"'>"+esc(n.type)+"</span>"+
    "<span class='pill' style='color:var(--muted);border-color:var(--border)'>"+esc(n.status)+"</span>"+
    "<div class='kv'>community <b>"+esc(com.name)+"</b></div>"+
    "<div class='kv'>importance <b>"+n.importance+"</b></div><div class='bar'><i style='width:"+
      Math.round(n.importance*100)+"%'></i></div>"+
    "<div class='kv'>confidence <b>"+n.confidence+"</b></div><div class='bar'><i style='width:"+
      Math.round(n.confidence*100)+"%'></i></div>"+
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
$("theme").onclick=function(){
  const light=document.documentElement.getAttribute("data-theme")==="light";
  document.documentElement.setAttribute("data-theme",light?"dark":"light");
  this.textContent=light?"Light":"Dark";
  applyEmphasis();
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
    if(f.indexOf("var(")===0)t.setAttribute("fill",light?"#1a1d27":"#e8eaf6");
    const s=t.getAttribute("stroke")||"";
    if(s.indexOf("var(")===0)t.setAttribute("stroke",light?"#ffffff":"#0f1117");
  });
  clone.querySelectorAll("line,path").forEach(p=>{
    const s=p.getAttribute("stroke")||"";
    if(s.indexOf("var(")===0)p.setAttribute("stroke",light?"#dfe3ef":"#2a2d3e");
  });
  const bg=document.createElementNS(NS,"rect");
  bg.setAttribute("width","100%");bg.setAttribute("height","100%");
  bg.setAttribute("fill",light?"#f7f8fc":"#0f1117");
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

for(let i=0;i<150;i++)step();
render();applyFilters();fit();requestAnimationFrame(tick);
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
            .replace("__COLORS__", _script_json(_TYPE_COLOR)))
    return html
