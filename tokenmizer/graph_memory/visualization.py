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
#
# REDESIGNED (2026-07-10 audit): the previous version was exactly the
# "generic Obsidian-style node soup" it claimed to beat —
#   - to_vis_json() exported `transitions` (the supersession history, the
#     ONE thing this product tracks that generic graph views don't), and
#     the HTML template never used them;
#   - superseded decisions were only lower-opacity circles;
#   - no filtering, no search, no export;
#   - it loaded D3 from a CDN, so the "self-contained" artifact broke
#     offline and in network-restricted demo environments.
# Now: zero external dependencies (hand-rolled force layout, ~2KB of JS),
# supersession arcs + a clickable decision-history timeline, per-type
# filter chips, active-only toggle, search, wheel-zoom/pan, PNG export.

_SHARE_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>TokenMizer — __SESSION__</title>
<style>
 :root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--txt:#c9d1d9;--dim:#8b949e;--hi:#5ee7c8;--red:#f87171}
 body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.45 'Segoe UI',system-ui,sans-serif;overflow:hidden}
 #hdr{position:fixed;top:0;left:0;right:340px;padding:14px 22px;display:flex;gap:18px;align-items:baseline;
      background:linear-gradient(#0d1117f0,#0d111700);z-index:3;pointer-events:none}
 #hdr b{font-size:18px;color:#e6edf3}
 #hdr .stat{color:var(--dim)}#hdr .stat i{color:var(--hi);font-style:normal;font-weight:600}
 #controls{position:fixed;top:54px;left:22px;z-index:3;display:flex;gap:8px;align-items:center;flex-wrap:wrap;max-width:55vw}
 #controls input[type=search]{background:var(--panel);border:1px solid var(--line);color:var(--txt);
      border-radius:6px;padding:5px 10px;font:inherit;font-size:12px;width:170px;outline:none}
 #controls input[type=search]:focus{border-color:var(--hi)}
 .btn{background:var(--panel);border:1px solid var(--line);color:var(--txt);border-radius:6px;
      padding:5px 11px;font-size:12px;cursor:pointer;user-select:none}
 .btn:hover{border-color:var(--hi)}
 .btn.on{border-color:var(--hi);color:var(--hi)}
 #chips{position:fixed;bottom:44px;left:22px;z-index:3;display:flex;flex-wrap:wrap;gap:8px;max-width:55vw}
 .chip{display:flex;align-items:center;gap:6px;color:var(--dim);font-size:12px;cursor:pointer;
      background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:3px 11px;user-select:none}
 .chip i{width:9px;height:9px;border-radius:50%;display:inline-block}
 .chip.off{opacity:0.35}
 #ftr{position:fixed;bottom:12px;left:22px;color:#484f58;font-size:12px;z-index:3}
 #ftr a{color:#7c6af7;text-decoration:none}
 #timeline{position:fixed;top:0;right:0;bottom:0;width:320px;background:var(--panel);
      border-left:1px solid var(--line);z-index:4;overflow-y:auto;padding:16px}
 #timeline h2{font-size:13px;text-transform:uppercase;letter-spacing:1px;color:var(--dim);margin:0 0 4px}
 #timeline .sub{font-size:12px;color:#484f58;margin-bottom:14px}
 .tr-item{border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:10px;
      cursor:pointer;transition:border-color .15s}
 .tr-item:hover,.tr-item.sel{border-color:var(--red)}
 .tr-item .old{color:var(--dim);text-decoration:line-through;font-size:12px}
 .tr-item .arrow{color:var(--red);margin:0 4px}
 .tr-item .new{color:#e6edf3;font-size:13px;font-weight:600}
 .tr-item .why{color:var(--dim);font-size:12px;margin-top:5px}
 .tr-item .meta{color:#484f58;font-size:11px;margin-top:4px}
 .empty{color:#484f58;font-size:13px;border:1px dashed var(--line);border-radius:8px;padding:16px;text-align:center}
 #tip{position:fixed;pointer-events:none;background:#1c2128;border:1px solid var(--line);border-radius:8px;
      padding:9px 12px;font-size:12px;max-width:300px;z-index:9;display:none;box-shadow:0 4px 16px #0009}
 #tip b{color:#e6edf3}#tip .st{color:var(--hi)}
 svg{cursor:grab;display:block}svg:active{cursor:grabbing}
</style></head><body>
<div id="hdr"><b>&#129504; __SESSION__</b>
 <span class="stat"><i>__NODES__</i> nodes</span>
 <span class="stat"><i>__EDGES__</i> edges</span>
 <span class="stat"><i>__DECISIONS__</i> decisions</span>
 <span class="stat"><i>__TRANSITIONS__</i> changed</span></div>
<div id="controls">
 <input id="search" type="search" placeholder="search nodes&#8230;"/>
 <span class="btn" id="activeOnly">Active only</span>
 <span class="btn" id="exportPng">&#11123; PNG</span>
 <span class="btn" id="reheat">&#8635; Layout</span>
</div>
<div id="chips"></div>
<div id="ftr">session memory graph &middot; <a href="https://github.com/Shweta-Mishra-ai/tokenmizer">TokenMizer</a> &middot; pip install tokenmizer</div>
<div id="timeline">
 <h2>Decision history</h2>
 <div class="sub">how this session's choices evolved</div>
 <div id="trList"></div>
</div>
<div id="tip"></div>
<script>
"use strict";
const DATA=__DATA__, COLOR=__COLORS__;
const INACTIVE=new Set(["superseded","archived","invalidated","modified"]);
// Clamp: innerWidth can be 0 in headless/embedded contexts, and a negative
// SVG width breaks rendering entirely.
const W=Math.max(640,(innerWidth||1280)-320), H=Math.max(480,innerHeight||800);
const NS="http://www.w3.org/2000/svg";
function el(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}

const svg=el("svg",{width:W,height:H,"font-family":"Segoe UI,system-ui,sans-serif"},document.body);
addEventListener("resize",()=>{
  svg.setAttribute("width",Math.max(640,innerWidth-320));
  svg.setAttribute("height",Math.max(480,innerHeight));
});
const defs=el("defs",{},svg);
const mk=el("marker",{id:"arr",viewBox:"0 -5 10 10",refX:18,refY:0,markerWidth:7,markerHeight:7,orient:"auto"},defs);
el("path",{d:"M0,-5L10,0L0,5",fill:"#f87171"},mk);
const vp=el("g",{},svg);           // viewport (zoom/pan)
const gE=el("g",{},vp), gT=el("g",{},vp), gN=el("g",{},vp); // edges, transitions, nodes

// ---- data prep -------------------------------------------------------------
const nodes=DATA.nodes, edges=DATA.edges, trans=DATA.transitions||[];
const byId={}; nodes.forEach(n=>byId[n.id]=n);
// seed positions: cluster by type around a circle
const types=[...new Set(nodes.map(n=>n.type))];
nodes.forEach((n,i)=>{
  const a=2*Math.PI*types.indexOf(n.type)/Math.max(types.length,1);
  n.x=W/2+Math.cos(a)*180+(Math.random()-0.5)*90;
  n.y=H/2+Math.sin(a)*180+(Math.random()-0.5)*90;
  n.vx=0;n.vy=0;
});
const links=edges.map(e=>({s:byId[e.source],t:byId[e.target],color:e.color}))
                 .filter(l=>l.s&&l.t);
const tlinks=trans.map(t=>({s:byId[t.from_id],t:byId[t.to_id],tr:t}))
                  .filter(l=>l.s&&l.t);

// ---- hand-rolled force simulation (no external libs) -----------------------
let alpha=1;
function tick(){
  for(let i=0;i<nodes.length;i++){ // pairwise repulsion (fine for <=200 nodes)
    const a=nodes[i];
    for(let j=i+1;j<nodes.length;j++){
      const b=nodes[j];
      let dx=b.x-a.x,dy=b.y-a.y,d2=dx*dx+dy*dy||1,d=Math.sqrt(d2);
      const f=Math.min(2200/d2,4)*alpha;
      dx/=d;dy/=d; a.vx-=dx*f;a.vy-=dy*f; b.vx+=dx*f;b.vy+=dy*f;
    }
    a.vx+=(W/2-a.x)*0.0018*alpha; a.vy+=(H/2-a.y)*0.0018*alpha; // centering
  }
  links.forEach(l=>{ // springs
    let dx=l.t.x-l.s.x,dy=l.t.y-l.s.y,d=Math.sqrt(dx*dx+dy*dy)||1;
    const f=(d-95)*0.012*alpha; dx/=d;dy/=d;
    l.s.vx+=dx*f*d*0.02;l.s.vy+=dy*f*d*0.02;
    l.t.vx-=dx*f*d*0.02;l.t.vy-=dy*f*d*0.02;
  });
  nodes.forEach(n=>{
    if(n.fixed)return;
    n.vx*=0.82;n.vy*=0.82; n.x+=n.vx;n.y+=n.vy;
  });
  alpha*=0.985;
  render();
  if(alpha>0.005)requestAnimationFrame(tick);
}

// ---- render ----------------------------------------------------------------
const eEls=links.map(l=>el("line",{stroke:l.color||"#30363d","stroke-opacity":0.4,"stroke-width":1.2},gE));
const tEls=tlinks.map(l=>{
  const p=el("path",{fill:"none",stroke:"#f87171","stroke-width":1.6,
    "stroke-dasharray":"6 4","stroke-opacity":0.85,"marker-end":"url(#arr)"},gT);
  p.dataset.trid=l.tr.id; return p;
});
const nEls=nodes.map(n=>{
  const g=el("g",{cursor:"pointer"},gN);
  const inactive=INACTIVE.has(n.status);
  const isActiveDecision=n.type==="decision"&&!inactive;
  if(isActiveDecision) // glow ring on ACTIVE decisions — the current truth
    el("circle",{r:(n.size||10)+5,fill:"none",stroke:COLOR[n.type]||"#8b949e",
      "stroke-opacity":0.5,"stroke-width":2},g);
  const c=el("circle",{r:n.size||10,fill:COLOR[n.type]||"#8b949e",
    "fill-opacity":inactive?0.22:(n.opacity??0.92)},g);
  if(inactive) // dashed ring marks dead branches — visually distinct, not just faded
    el("circle",{r:(n.size||10)+3,fill:"none",stroke:"#8b949e",
      "stroke-dasharray":"3 3","stroke-opacity":0.55,"stroke-width":1},g);
  if(n.status==="invalidated")
    el("circle",{r:(n.size||10)+3,fill:"none",stroke:"#f87171",
      "stroke-dasharray":"2 2","stroke-width":1.4},g);
  const lbl=el("text",{dx:(n.size||10)+5,dy:4,fill:inactive?"#6e7681":"#c9d1d9",
    "font-size":11,"paint-order":"stroke",stroke:"#0d1117","stroke-width":3},g);
  lbl.textContent=n.label.length>34?n.label.slice(0,32)+"\\u2026":n.label;
  if(inactive)lbl.setAttribute("text-decoration","line-through");
  g._n=n; n._el=g;
  // drag
  let drag=false;
  g.addEventListener("pointerdown",ev=>{drag=true;n.fixed=true;g.setPointerCapture(ev.pointerId);ev.stopPropagation()});
  g.addEventListener("pointermove",ev=>{if(!drag)return;
    const m=pt(ev);n.x=m.x;n.y=m.y;alpha=Math.max(alpha,0.08);
    if(alpha<=0.09)requestAnimationFrame(tick);render()});
  g.addEventListener("pointerup",()=>{drag=false;n.fixed=false});
  // tooltip
  g.addEventListener("pointerenter",ev=>{
    tip.style.display="block";
    tip.innerHTML="<b>"+esc(n.label)+"</b><br><span class='st'>"+n.type+" &middot; "+n.status+
      "</span>"+(n.summary?"<br>"+esc(n.summary):"")+
      "<br><span style='color:#484f58'>importance "+n.importance+" &middot; confidence "+n.confidence+"</span>";
  });
  g.addEventListener("pointermove",ev=>{tip.style.left=(ev.clientX+14)+"px";tip.style.top=(ev.clientY+10)+"px"});
  g.addEventListener("pointerleave",()=>tip.style.display="none");
  return g;
});
function render(){
  links.forEach((l,i)=>{const e=eEls[i];
    e.setAttribute("x1",l.s.x);e.setAttribute("y1",l.s.y);
    e.setAttribute("x2",l.t.x);e.setAttribute("y2",l.t.y)});
  tlinks.forEach((l,i)=>{
    const mx=(l.s.x+l.t.x)/2, my=(l.s.y+l.t.y)/2;
    const dx=l.t.x-l.s.x, dy=l.t.y-l.s.y, d=Math.sqrt(dx*dx+dy*dy)||1;
    tEls[i].setAttribute("d","M"+l.s.x+","+l.s.y+" Q"+(mx-dy/d*40)+","+(my+dx/d*40)+" "+l.t.x+","+l.t.y)});
  nodes.forEach(n=>n._el.setAttribute("transform","translate("+n.x+","+n.y+")"));
}

// ---- zoom / pan ------------------------------------------------------------
let z={k:1,x:0,y:0};
function applyZ(){vp.setAttribute("transform","translate("+z.x+","+z.y+") scale("+z.k+")")}
function pt(ev){const r=svg.getBoundingClientRect();
  return {x:(ev.clientX-r.left-z.x)/z.k, y:(ev.clientY-r.top-z.y)/z.k}}
svg.addEventListener("wheel",ev=>{ev.preventDefault();
  const s=ev.deltaY<0?1.15:0.87, nk=Math.min(4,Math.max(0.2,z.k*s));
  const r=svg.getBoundingClientRect(),mx=ev.clientX-r.left,my=ev.clientY-r.top;
  z.x=mx-(mx-z.x)*(nk/z.k); z.y=my-(my-z.y)*(nk/z.k); z.k=nk; applyZ();
},{passive:false});
let panning=false,px=0,py=0;
svg.addEventListener("pointerdown",ev=>{if(ev.target===svg||ev.target===vp){panning=true;px=ev.clientX;py=ev.clientY}});
addEventListener("pointermove",ev=>{if(!panning)return;
  z.x+=ev.clientX-px;z.y+=ev.clientY-py;px=ev.clientX;py=ev.clientY;applyZ()});
addEventListener("pointerup",()=>panning=false);

// ---- filters ---------------------------------------------------------------
const hidden=new Set(); let activeOnly=false, q="";
function visible(n){
  if(hidden.has(n.type))return false;
  if(activeOnly&&INACTIVE.has(n.status))return false;
  return true;
}
function applyFilters(){
  nodes.forEach(n=>{
    const v=visible(n);
    const match=!q||n.label.toLowerCase().includes(q)||(n.summary||"").toLowerCase().includes(q);
    n._el.style.display=v?"":"none";
    n._el.style.opacity=match?1:0.12;
  });
  links.forEach((l,i)=>eEls[i].style.display=(visible(l.s)&&visible(l.t))?"":"none");
  tlinks.forEach((l,i)=>tEls[i].style.display=(visible(l.s)&&visible(l.t))?"":"none");
}
const chips=document.getElementById("chips");
types.forEach(t=>{
  const c=document.createElement("span");c.className="chip";
  c.innerHTML="<i style='background:"+(COLOR[t]||"#8b949e")+"'></i>"+t;
  c.onclick=()=>{hidden.has(t)?hidden.delete(t):hidden.add(t);
    c.classList.toggle("off",hidden.has(t));applyFilters()};
  chips.appendChild(c);
});
document.getElementById("activeOnly").onclick=function(){
  activeOnly=!activeOnly;this.classList.toggle("on",activeOnly);applyFilters()};
document.getElementById("search").addEventListener("input",function(){
  q=this.value.trim().toLowerCase();applyFilters()});
document.getElementById("reheat").onclick=()=>{alpha=1;requestAnimationFrame(tick)};

// ---- decision-history timeline ----------------------------------------------
const trList=document.getElementById("trList");
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
if(!trans.length){
  trList.innerHTML="<div class='empty'>No decision changes yet.<br>When a decision is superseded, the old&#8594;new story appears here.</div>";
}else{
  [...trans].sort((a,b)=>(b.timestamp||0)-(a.timestamp||0)).forEach(t=>{
    const d=document.createElement("div");d.className="tr-item";
    const when=t.timestamp?new Date(t.timestamp*1000).toLocaleString():"";
    d.innerHTML="<span class='old'>"+esc(t.from_label)+"</span><span class='arrow'>&#8594;</span>"+
      "<span class='new'>"+esc(t.to_label)+"</span>"+
      (t.reason?"<div class='why'>"+esc(t.reason)+"</div>":"")+
      "<div class='meta'>"+esc(t.trigger||"superseded")+(when?" &middot; "+when:"")+"</div>";
    d.onclick=()=>{
      document.querySelectorAll(".tr-item.sel").forEach(x=>x.classList.remove("sel"));
      d.classList.add("sel");
      const a=byId[t.from_id],b=byId[t.to_id];
      nodes.forEach(n=>n._el.style.opacity=(n===a||n===b)?1:0.12);
      if(b){ // center the view on the new decision
        z.x=W/2-b.x*z.k; z.y=H/2-b.y*z.k; applyZ();
      }
    };
    trList.appendChild(d);
  });
}
document.addEventListener("keydown",e=>{if(e.key==="Escape"){
  document.querySelectorAll(".tr-item.sel").forEach(x=>x.classList.remove("sel"));
  q="";document.getElementById("search").value="";applyFilters()}});

// ---- PNG export --------------------------------------------------------------
document.getElementById("exportPng").onclick=()=>{
  const clone=svg.cloneNode(true);
  clone.setAttribute("xmlns",NS);
  const bg=document.createElementNS(NS,"rect");
  bg.setAttribute("width","100%");bg.setAttribute("height","100%");bg.setAttribute("fill","#0d1117");
  clone.insertBefore(bg,clone.firstChild);
  const blob=new Blob([new XMLSerializer().serializeToString(clone)],{type:"image/svg+xml"});
  const url=URL.createObjectURL(blob), img=new Image();
  img.onload=()=>{
    const cv=document.createElement("canvas");cv.width=W*2;cv.height=H*2;
    const ctx=cv.getContext("2d");ctx.scale(2,2);ctx.drawImage(img,0,0);
    URL.revokeObjectURL(url);
    const a=document.createElement("a");
    a.download="tokenmizer-__SESSION__.png";a.href=cv.toDataURL("image/png");a.click();
  };
  img.src=url;
};

render();applyFilters();requestAnimationFrame(tick);
</script></body></html>
"""


def to_share_html(graph: "GraphMemory") -> str:
    """Self-contained dark interactive graph HTML — open in any browser,
    zero network dependencies (works offline / air-gapped demo).

    What it shows that a generic graph view doesn't:
      - decision supersession arcs (old → new, dashed red, arrowhead)
      - a clickable "Decision history" timeline panel (old label struck
        through → new label, with trigger + reason + timestamp); clicking
        an entry spotlights the two nodes and centers the view
      - active decisions get a glow ring; superseded/archived get a dashed
        ring + strikethrough label; invalidated get a red dashed ring
      - per-type filter chips, "Active only" toggle, text search,
        wheel-zoom/pan, and one-click PNG export
    """
    import json as _json

    vis = graph.to_vis_json()
    nodes = vis.get("nodes", [])
    edges = vis.get("edges", [])
    transitions = vis.get("transitions", [])
    decisions = sum(1 for n in nodes if n.get("type") == "decision")
    html = (_SHARE_HTML_TEMPLATE
            .replace("__SESSION__", graph.session_id)
            .replace("__NODES__", str(len(nodes)))
            .replace("__EDGES__", str(len(edges)))
            .replace("__DECISIONS__", str(decisions))
            .replace("__TRANSITIONS__", str(len(transitions)))
            .replace("__DATA__", _json.dumps({
                "nodes": nodes, "edges": edges, "transitions": transitions,
            }))
            .replace("__COLORS__", _json.dumps(_TYPE_COLOR)))
    return html
