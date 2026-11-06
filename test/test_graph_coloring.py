"""Check proper coloring, first-fit behavior, and smallest-last degree bounds."""

from itertools import combinations, product

import pytest

from src.graph import Graph
from src.graph_coloring import (
    greedy_coloring,
    smallest_last_coloring,
    smallest_last_ordering,
)


def assert_proper_coloring(graph, colors):
    """Assert complete vertex coverage, legal edges, and consecutive colors."""
    assert colors.keys() == set(graph.vertices)
    assert all(isinstance(color, int) and color >= 0 for color in colors.values())
    assert all(colors[u] != colors[v] for u, v in graph.edges)
    assert set(colors.values()) == set(range(len(set(colors.values()))))


@pytest.mark.parametrize("algorithm", [greedy_coloring, smallest_last_coloring])
@pytest.mark.parametrize("vertices", [[], [-7, 0, 99]])
def test_edgeless_graph_uses_one_color_per_nonempty_graph(algorithm, vertices):
    """Check isolates share color zero and an empty graph uses no colors."""
    assert algorithm(Graph(vertices=vertices)) == {vertex: 0 for vertex in vertices}


def test_greedy_follows_input_order_and_chooses_lowest_legal_color():
    """Verify the exact first-fit coloring for a deliberately awkward path order."""
    graph = Graph([(1, 2), (2, 3), (3, 4)], vertices=[1, 4, 2, 3, 5])

    assert greedy_coloring(graph) == {1: 0, 4: 0, 2: 1, 3: 2, 5: 0}


def test_smallest_last_improves_badly_ordered_path():
    """Check smallest-last ordering reduces an awkward path from three colors to two."""
    graph = Graph([(1, 2), (2, 3), (3, 4)], vertices=[1, 4, 2, 3, 5])

    colors = smallest_last_coloring(graph)

    assert_proper_coloring(graph, colors)
    assert len(set(greedy_coloring(graph).values())) == 3
    assert len(set(colors.values())) == 2
    assert list(colors) == smallest_last_ordering(graph)


@pytest.mark.parametrize("algorithm", [greedy_coloring, smallest_last_coloring])
@pytest.mark.parametrize("size", [1, 4, 8])
def test_complete_graph_needs_a_different_color_for_each_vertex(algorithm, size):
    """Verify that each vertex of a complete graph receives a distinct color."""
    graph = Graph(combinations(range(size), 2), vertices=range(size))

    colors = algorithm(graph)

    assert_proper_coloring(graph, colors)
    assert len(set(colors.values())) == size


@pytest.mark.parametrize("algorithm", [greedy_coloring, smallest_last_coloring])
@pytest.mark.parametrize("size, expected", [(4, 2), (5, 3)])
def test_even_and_odd_cycles(algorithm, size, expected):
    """Check even cycles use two colors and odd cycles use three."""
    graph = Graph((vertex, (vertex + 1) % size) for vertex in range(size))

    colors = algorithm(graph)

    assert_proper_coloring(graph, colors)
    assert len(set(colors.values())) == expected


@pytest.mark.parametrize("algorithm", [greedy_coloring, smallest_last_coloring])
def test_disconnected_graph_preserves_labels_and_reuses_colors(algorithm):
    """Check arbitrary labels and color reuse across disconnected components."""
    graph = Graph(
        [(-7, 0), (0, 99), (99, -7), (100_000, 900_000)],
        vertices=[42],
    )

    colors = algorithm(graph)

    assert_proper_coloring(graph, colors)
    assert len(set(colors.values())) == 3
    assert colors[42] == 0


def test_all_five_vertex_graphs_have_proper_colorings_and_degree_bounds():
    """Exhaustively check proper coloring, minimum-degree removals, and bounds."""
    possible_edges = tuple(combinations(range(5), 2))
    for included in product((False, True), repeat=len(possible_edges)):
        graph = Graph(
            (edge for edge, keep in zip(possible_edges, included) if keep),
            vertices=range(5),
        )
        order = smallest_last_ordering(graph)
        assert len(order) == len(graph)
        assert set(order) == set(graph.vertices)

        # Independently check every removal against degrees in the remaining graph.
        remaining = set(graph.vertices)
        degeneracy = 0
        for vertex in reversed(order):
            degree = len(graph.neighbors(vertex) & remaining)
            assert degree == min(len(graph.neighbors(v) & remaining) for v in remaining)
            degeneracy = max(degeneracy, degree)
            remaining.remove(vertex)

        for algorithm, bound in (
            (greedy_coloring, max(map(graph.degree, graph.vertices)) + 1),
            (smallest_last_coloring, degeneracy + 1),
        ):
            colors = algorithm(graph)
            assert_proper_coloring(graph, colors)
            assert len(set(colors.values())) <= bound


def test_empty_graph_has_empty_smallest_last_order():
    """Verify that smallest-last ordering has no removals for an empty graph."""
    assert smallest_last_ordering(Graph()) == []


@pytest.mark.parametrize("algorithm", [greedy_coloring, smallest_last_coloring])
def test_large_star_including_isolates(algorithm):
    """Check a large star remains properly colored with two colors and an isolate."""
    graph = Graph(((0, leaf) for leaf in range(1, 1534)), vertices=range(1535))

    colors = algorithm(graph)

    assert_proper_coloring(graph, colors)
    assert len(set(colors.values())) == 2


@pytest.mark.parametrize(
    "algorithm", [greedy_coloring, smallest_last_coloring, smallest_last_ordering]
)
def test_algorithms_are_repeatable_and_leave_graph_unchanged(algorithm):
    """Check deterministic results and unchanged storage after result mutation."""
    graph = Graph([(1, 2), (2, 3), (3, 1), (4, 5)], vertices=range(1, 7))
    vertices, edges = graph.vertices, graph.edges
    neighbors = {vertex: graph.neighbors(vertex) for vertex in vertices}

    first = algorithm(graph)
    assert algorithm(graph) == first
    first.clear()

    assert graph.vertices == vertices
    assert graph.edges == edges
    assert {vertex: graph.neighbors(vertex) for vertex in vertices} == neighbors
