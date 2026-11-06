"""Unit tests for graph storage and DIMACS loading."""

from itertools import combinations
from random import getstate

import pytest

from src.graph import Graph


def test_empty_graph():
    """Verify that a default graph has no vertices or edges."""
    graph = Graph()

    assert len(graph) == 0
    assert graph.vertices == ()
    assert graph.edges == ()


def test_adjacency_is_undirected_and_includes_isolated_vertices():
    """Check symmetric neighbor sets, degrees, and explicit isolated vertices."""
    graph = Graph([(1, 2), (2, 3)], vertices=[1, 2, 3, 4])

    assert graph.vertices == (1, 2, 3, 4)
    assert len(graph) == 4
    assert graph.edges == ((1, 2), (2, 3))
    assert graph.neighbors(1) == {2}
    assert graph.neighbors(2) == {1, 3}
    assert graph.neighbors(3) == {2}
    assert graph.neighbors(4) == set()
    assert [graph.degree(vertex) for vertex in graph.vertices] == [1, 2, 1, 0]


def test_endpoints_are_added_and_duplicates_are_ignored():
    """Check insertion order and deduplication of vertices and undirected edges."""
    graph = Graph([(3, 1), (1, 3), (3, 1), (1, 2)], vertices=[4, 4])

    assert graph.vertices == (4, 3, 1, 2)
    assert graph.edges == ((3, 1), (1, 2))
    assert graph.degree(3) == 1
    assert graph.degree(1) == 2


def test_graph_does_not_share_mutable_data_with_caller():
    """Ensure later mutations to input collections cannot alter a graph."""
    edges = [[1, 2]]
    vertices = [3]
    graph = Graph(edges, vertices)
    edges[0][0] = 9
    vertices.append(4)

    assert graph.edges == ((1, 2),)
    assert graph.vertices == (3, 1, 2)
    assert isinstance(graph.neighbors(1), frozenset)


def test_accepts_generators():
    """Verify that edge and vertex inputs may be single-pass generators."""
    graph = Graph(((v, v + 1) for v in range(3)), (v for v in range(5)))

    assert graph.edges == ((0, 1), (1, 2), (2, 3))
    assert graph.vertices == (0, 1, 2, 3, 4)


def test_self_loops_are_rejected():
    """Require a clear error when an edge joins a vertex to itself."""
    with pytest.raises(ValueError, match="Self-loops"):
        Graph([(1, 1)])


def test_unknown_vertex_raises_key_error():
    """Verify that neighbor lookup rejects labels absent from the graph."""
    with pytest.raises(KeyError):
        Graph().neighbors(1)


def test_seeded_shuffle_preserves_graph_and_global_random_state():
    """Check seeded order changes while preserving adjacency and random state."""
    graph = Graph(combinations(range(6), 2), vertices=[99, *range(6)])
    vertices, edges = graph.vertices, graph.edges
    neighbors = {vertex: graph.neighbors(vertex) for vertex in vertices}
    state = getstate()

    first = graph.shuffled(seed=42)
    repeated = graph.shuffled(seed=42)
    other = graph.shuffled(seed=43)

    assert first is not graph
    assert first.vertices == repeated.vertices
    assert first.edges == repeated.edges
    assert first.vertices != other.vertices
    assert first.edges != other.edges
    assert first.vertices != vertices
    assert first.edges != edges
    assert set(first.vertices) == set(vertices)
    assert set(first.edges) == set(edges)
    assert {vertex: first.neighbors(vertex) for vertex in vertices} == neighbors
    assert first.degree(99) == 0
    assert graph.vertices == vertices
    assert graph.edges == edges
    assert {vertex: graph.neighbors(vertex) for vertex in vertices} == neighbors
    assert getstate() == state


@pytest.mark.parametrize("vertices", [[], [-7, 0, 99]])
def test_shuffle_handles_empty_and_edgeless_graphs(vertices):
    """Ensure shuffling preserves all isolates even when there are no edges."""
    shuffled = Graph(vertices=vertices).shuffled(seed=0)

    assert set(shuffled.vertices) == set(vertices)
    assert shuffled.edges == ()
    assert all(shuffled.degree(vertex) == 0 for vertex in vertices)


def test_load_dimacs_preserves_labels_and_isolated_vertices(tmp_path):
    """Read DIMACS comments and whitespace without losing declared isolates."""
    path = tmp_path / "example.mis"
    path.write_bytes(b"c example\r\n\r\np edge 4 2   \r\ne 1 2\r\nc comment\r\ne 3 2\r\n")

    graph = Graph.from_dimacs(path)

    assert graph.vertices == (1, 2, 3, 4)
    assert graph.edges == ((1, 2), (3, 2))
    assert graph.neighbors(2) == {1, 3}
    assert graph.degree(4) == 0


@pytest.mark.parametrize("num_vertices", [0, 3])
def test_load_dimacs_without_edges(tmp_path, num_vertices):
    """Verify that a DIMACS header alone creates every declared vertex."""
    path = tmp_path / "empty.clq"
    path.write_text(f"p edge {num_vertices} 0\n", encoding="utf-8")

    graph = Graph.from_dimacs(str(path))

    assert graph.vertices == tuple(range(1, num_vertices + 1))
    assert graph.edges == ()


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "c comments only\n",
        "e 1 2\np edge 2 1\n",
        "p edge 2\n",
        "p other 2 0\n",
        "p edge two 0\n",
        "p edge -1 0\n",
        "p edge 2 -1\n",
        "p edge 2 0\np edge 2 0\n",
        "p edge 2 1\ne 0 1\n",
        "p edge 2 1\ne 1 3\n",
        "p edge 2 1\ne 1\n",
        "p edge 2 1\ne 1 two\n",
        "p edge 2 1\ne 1 2 extra\n",
        "p edge 2 1\ne 1 1\n",
        "p edge 2 2\ne 1 2\ne 2 1\n",
        "p edge 2 1\n",
        "p edge 2 0\ne 1 2\n",
        "p edge 2 0\nx 1 2\n",
    ],
)
def test_invalid_dimacs_is_rejected(tmp_path, contents):
    """Reject malformed records, invalid endpoints, and inconsistent edge counts."""
    path = tmp_path / "invalid.mis"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError):
        Graph.from_dimacs(path)
