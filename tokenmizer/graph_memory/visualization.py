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
