"""Check cover validity and approximation bounds against exact small optima."""

from itertools import combinations, product
from random import getstate
from types import SimpleNamespace

import pytest

from src.graph import Graph
from src.vertex_cover import approx_vertex_cover, randomized_vertex_cover


def is_cover(graph, cover):
    """Return whether graph vertices in ``cover`` touch every edge."""
    return cover <= set(graph.vertices) and all(
        u in cover or v in cover for u, v in graph.edges
    )


def optimal_cover_size(graph):
    """Return the exact minimum cover size by enumerating vertex subsets."""
    for size in range(len(graph) + 1):
        if any(is_cover(graph, set(nodes)) for nodes in combinations(graph.vertices, size)):
            return size


def small_graphs():
    """Yield all 64 simple graphs on four labeled vertices, including isolates."""
    possible_edges = tuple(combinations(range(4), 2))
    for included in product((False, True), repeat=len(possible_edges)):
        yield Graph(
            (edge for edge, keep in zip(possible_edges, included) if keep),
            vertices=range(4),
        )


@pytest.mark.parametrize("algorithm", [approx_vertex_cover, randomized_vertex_cover])
@pytest.mark.parametrize("vertices", [[], [1, 2, 3]])
def test_edgeless_graph_has_empty_cover(algorithm, vertices):
    """Verify that neither algorithm selects vertices when there are no edges."""
    assert algorithm(Graph(vertices=vertices)) == set()


@pytest.mark.parametrize("algorithm", [approx_vertex_cover, randomized_vertex_cover])
@pytest.mark.parametrize(
    "vertices", [(-7, 0, 12, 99), (100_000, 17, 900_000, 42)]
)
def test_cover_preserves_original_labels(algorithm, vertices):
    """Check triangle covers retain arbitrary labels and exclude an isolate."""
    u, v, w, isolated = vertices
    graph = Graph([(u, v), (v, w), (w, u)], vertices=vertices)

    cover = algorithm(graph)

    assert isinstance(cover, set)
    assert is_cover(graph, cover)
    assert len(cover) == 2
    assert isolated not in cover


@pytest.mark.parametrize(
    "algorithm, vertices_per_edge",
    [(approx_vertex_cover, 2), (randomized_vertex_cover, 1)],
)
@pytest.mark.parametrize("num_pairs", [32, 767])
def test_cover_with_many_vertices(algorithm, vertices_per_edge, num_pairs):
    """Check cover sizes on many disjoint edges with an additional isolate."""
    # Disjoint edges exercise large covers and leave one isolated vertex.
    graph = Graph(
        [(v, v + 1) for v in range(0, 2 * num_pairs, 2)],
        vertices=range(2 * num_pairs + 1),
    )

    cover = algorithm(graph)

    assert is_cover(graph, cover)
    assert len(cover) == vertices_per_edge * num_pairs
    assert 2 * num_pairs not in cover


def test_approximation_selects_both_endpoints_of_uncovered_edges():
    """Verify the deterministic edge scan skips edges already covered."""
    graph = Graph([(1, 2), (2, 3), (3, 4), (4, 5)])

    assert approx_vertex_cover(graph) == {1, 2, 3, 4}


def test_approximation_is_a_cover_within_twice_optimum():
    """Check validity and the factor-two bound on every four-vertex graph."""
    for graph in small_graphs():
        cover = approx_vertex_cover(graph)

        assert is_cover(graph, cover), graph.edges
        assert len(cover) <= 2 * optimal_cover_size(graph), graph.edges


def test_randomized_expected_bound_for_all_small_graphs(monkeypatch):
    """Check the exact expected factor-two bound by enumerating random choices."""
    # Enumerating equally likely coin strings gives the exact expectation.
    # Unused suffixes account for paths that finish with fewer random choices.
    for graph in small_graphs():
        total_size = 0
        for coins in product((0, 1), repeat=len(graph.edges)):
            flips = iter(coins)
            rng = SimpleNamespace(choice=lambda edge: edge[next(flips)])
            monkeypatch.setattr("src.vertex_cover.Random", lambda seed: rng)
            cover = randomized_vertex_cover(graph)

            assert is_cover(graph, cover), graph.edges
            total_size += len(cover)

        outcomes = 2 ** len(graph.edges)
        assert total_size <= 2 * optimal_cover_size(graph) * outcomes, graph.edges


@pytest.mark.parametrize("endpoint", [0, 1])
def test_randomized_algorithm_can_choose_either_endpoint(monkeypatch, endpoint):
    """Force each possible random endpoint choice on a single edge."""
    rng = SimpleNamespace(choice=lambda edge: edge[endpoint])
    monkeypatch.setattr("src.vertex_cover.Random", lambda seed: rng)

    assert randomized_vertex_cover(Graph([(1, 2)])) == {(1, 2)[endpoint]}


def test_randomized_bound_is_in_expectation_not_for_every_run(monkeypatch):
    """Demonstrate a valid star cover whose size exceeds twice the optimum."""
    graph = Graph([(0, leaf) for leaf in range(1, 5)])
    rng = SimpleNamespace(choice=lambda edge: edge[1])
    monkeypatch.setattr("src.vertex_cover.Random", lambda seed: rng)

    cover = randomized_vertex_cover(graph)

    assert cover == {1, 2, 3, 4}
    assert is_cover(graph, cover)
    assert len(cover) > 2 * optimal_cover_size(graph)


def test_randomized_seed_is_reproducible_and_does_not_change_global_state():
    """Check repeatable endpoint choices without modifying global random state."""
    graph = Graph(combinations(range(8), 2))
    state = getstate()
    expected = randomized_vertex_cover(graph, seed=42)
    randomized_vertex_cover(graph, seed=7)

    assert randomized_vertex_cover(graph, seed=42) == expected
    assert getstate() == state


def test_randomized_algorithm_with_real_generators():
    """Check cover validity across small graphs using several real RNG seeds."""
    for graph in small_graphs():
        for seed in range(8):
            assert is_cover(graph, randomized_vertex_cover(graph, seed=seed))


@pytest.mark.parametrize("algorithm", [approx_vertex_cover, randomized_vertex_cover])
def test_algorithms_leave_the_graph_unchanged(algorithm):
    """Ensure cover construction and result mutation preserve graph storage."""
    graph = Graph([(1, 2), (2, 3), (3, 1), (4, 5)], vertices=range(1, 7))
    vertices = graph.vertices
    edges = graph.edges
    neighbors = {vertex: graph.neighbors(vertex) for vertex in vertices}

    cover = algorithm(graph)
    cover.clear()

    assert graph.vertices == vertices
    assert graph.edges == edges
    assert {vertex: graph.neighbors(vertex) for vertex in vertices} == neighbors
