"""graph_memory/communities.py — deterministic label propagation used by
the graph page to color and group nodes."""
from __future__ import annotations

import random

from tokenmizer.graph_memory.communities import detect_communities

# Two triangles joined by a single bridge edge: the canonical two-community
# graph. Node ids are deliberately not in a-b-c order so the result cannot
# depend on insertion order.
TRIANGLE_A = ["a1", "a2", "a3"]
TRIANGLE_B = ["b1", "b2", "b3"]
EDGES = [
    ("a1", "a2", 1.0), ("a2", "a3", 1.0), ("a3", "a1", 1.0),
    ("b1", "b2", 1.0), ("b2", "b3", 1.0), ("b3", "b1", 1.0),
    ("a1", "b1", 1.0),
]


def test_two_triangles_form_two_communities():
    result = detect_communities(TRIANGLE_A + TRIANGLE_B, EDGES)
    a = {result[n] for n in TRIANGLE_A}
    b = {result[n] for n in TRIANGLE_B}
    assert len(a) == 1, "nodes in the same triangle must share a community"
    assert len(b) == 1
    assert a != b, "the two triangles must be separate communities"


def test_community_indices_are_dense_and_ordered_by_size():
    result = detect_communities(TRIANGLE_A + TRIANGLE_B + ["lone"], EDGES)
    assert set(result.values()) == {0, 1, 2}


def test_isolated_nodes_share_one_trailing_community():
    result = detect_communities(TRIANGLE_A + ["x", "y", "z"], EDGES[:3])
    assert result["x"] == result["y"] == result["z"]
    assert result["x"] > result["a1"], "singletons go last, after real clusters"


def test_result_is_deterministic_across_input_orders():
    ids = TRIANGLE_A + TRIANGLE_B + ["x", "y"]
    first = detect_communities(ids, EDGES)
    for seed in range(5):
        rng = random.Random(seed)
        shuffled_ids = ids[:]
        shuffled_edges = EDGES[:]
        rng.shuffle(shuffled_ids)
        rng.shuffle(shuffled_edges)
        assert detect_communities(shuffled_ids, shuffled_edges) == first


def test_empty_graph():
    assert detect_communities([], []) == {}


def test_edges_to_unknown_nodes_are_ignored():
    result = detect_communities(["a", "b"], [("a", "b", 1.0), ("a", "ghost", 1.0)])
    assert set(result) == {"a", "b"}
    assert result["a"] == result["b"]
