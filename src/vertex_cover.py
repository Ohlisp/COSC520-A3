"""Vertex-cover approximations using a dictionary of selection flags.

Both algorithms initialize flags for all vertices, scan each edge once, and
collect the selected vertices into a set. With average O(1) dictionary
operations, they take O(|V| + |E|) expected time and O(|V|) additional space.
"""

from random import Random

from .graph import Graph


def approx_vertex_cover(graph: Graph) -> set[int]:
    """Return a vertex cover of size at most twice the optimum.

    Scan edges in input order, choosing both endpoints of each uncovered edge.
    Skipping edges incident to chosen vertices implements the deletion step
    without modifying the graph. Return a set of original vertex labels; an
    edgeless graph returns an empty set.
    """
    selected = dict.fromkeys(graph.vertices, False)
    for u, v in graph.edges:
        if not (selected[u] | selected[v]):
            # These uncovered edges form a matching, giving the factor-two bound.
            selected[u] = True
            selected[v] = True
    return {vertex for vertex, is_selected in selected.items() if is_selected}


def randomized_vertex_cover(graph: Graph, seed: int | None = None) -> set[int]:
    """Return a cover whose expected size is at most twice the optimum.

    Choose one endpoint uniformly at random for each uncovered edge, scanning
    edges in input order. Individual runs can exceed the factor of two.
    A local generator allows seeded runs without changing global random state.
    The graph is unchanged; the result contains the original vertex labels.

    Args:
        graph: Undirected graph whose edges must all be covered.
        seed: Random seed, or None to seed from the generator's default source.

    Returns:
        Selected vertex labels, or an empty set for an edgeless graph.
    """
    rng = Random(seed)
    selected = dict.fromkeys(graph.vertices, False)
    for u, v in graph.edges:
        if not (selected[u] | selected[v]):
            selected[rng.choice((u, v))] = True
    return {vertex for vertex, is_selected in selected.items() if is_selected}
