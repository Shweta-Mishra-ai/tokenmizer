"""
Community detection for the graph page: which nodes belong together.

Label propagation, kept deterministic on purpose. The page colors nodes by
community and lists the groups in a side panel, so the same graph must
produce the same grouping on every render, on every platform. Nothing
here reads a clock, a random source, or dict insertion order: nodes are
processed in sorted-id order, ties resolve to the smallest label, and
the final indices are assigned by (size desc, smallest member id).

Graphs here are small (the auto-prune cap is 200 nodes), so the O(iter x
edges) sweep costs nothing measurable.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

# ponytail: label propagation, O(iter * E). Swap for Louvain only if graphs
# ever exceed the 200-node prune cap and the groupings look wrong at that
# size — not before, since this is deterministic and Louvain is not.
DEFAULT_MAX_ITER = 20


def detect_communities(
    node_ids: Iterable[str],
    edges: Iterable[tuple[str, str, float]],
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict[str, int]:
    """Map every node id to a community index.

    Indices are dense, 0..k-1, ordered by community size (largest first),
    ties broken by the smallest member id. Every node with no neighbors
    is folded into one trailing group so the page can label them
    "Unclustered" instead of showing dozens of one-node communities.
    Edges that name an unknown node are ignored.
    """
    ordered = sorted(set(node_ids))
    if not ordered:
        return {}
    known = set(ordered)

    # Undirected weighted adjacency; the graph's edges are directed but
    # "belongs together" is symmetric.
    adjacency: dict[str, dict[str, float]] = defaultdict(dict)
    for source, target, weight in edges:
        if source not in known or target not in known or source == target:
            continue
        adjacency[source][target] = adjacency[source].get(target, 0.0) + weight
        adjacency[target][source] = adjacency[target].get(source, 0.0) + weight

    label = {node: index for index, node in enumerate(ordered)}

    # Low-degree nodes first. Members of a tight cluster consolidate onto
    # one label before the hubs and bridge nodes that connect clusters get
    # to choose, so a bridge sees "two neighbors already agree" rather
    # than three distinct labels it has to break a tie between. With
    # every node seeded to its own label, id order alone lets the bridge
    # pick the foreign cluster's label and the two clusters collapse into
    # one — the two-triangle case in the tests.
    sweep = sorted(ordered, key=lambda node: (len(adjacency.get(node, ())), node))

    for _ in range(max_iter):
        changed = False
        for node in sweep:
            neighbors = adjacency.get(node)
            if not neighbors:
                continue
            weight_by_label: dict[int, float] = defaultdict(float)
            for neighbor, weight in neighbors.items():
                weight_by_label[label[neighbor]] += weight
            top = max(weight_by_label.values())
            tied = sorted(lab for lab, w in weight_by_label.items() if w == top)
            # Keep the current label when it is among the best (stability,
            # no oscillation); otherwise the smallest best label wins.
            best = label[node] if label[node] in tied else tied[0]
            if best != label[node]:
                label[node] = best
                changed = True
        if not changed:
            break

    members: dict[int, list[str]] = defaultdict(list)
    for node in ordered:
        members[label[node]].append(node)

    clustered = [group for group in members.values() if len(group) > 1]
    singletons = [node for group in members.values() if len(group) == 1 for node in group]
    clustered.sort(key=lambda group: (-len(group), group[0]))

    result: dict[str, int] = {}
    for index, group in enumerate(clustered):
        for node in group:
            result[node] = index
    for node in singletons:
        result[node] = len(clustered)
    return result
