"""Greedy graph coloring in input order or smallest-last order."""

from collections.abc import Iterable

from .graph import Graph


def _greedy_coloring(graph: Graph, order: Iterable[int]) -> dict[int, int]:
    """Assign the lowest legal nonnegative color to each vertex in ``order``.

    The caller supplies each graph vertex exactly once. Only already colored
    neighbors forbid colors; dictionary insertion order follows ``order``.
    Return vertex labels mapped to colors without changing the graph.
    """
    colors = {}
    for vertex in order:
        forbidden = {
            colors[neighbor]
            for neighbor in graph.neighbors(vertex)
            if neighbor in colors
        }
        color = 0
        while color in forbidden:
            color += 1
        colors[vertex] = color
    return colors


def greedy_coloring(graph: Graph) -> dict[int, int]:
    """Assign each vertex the lowest legal color, following graph input order.

    Return original vertex labels mapped to colors starting at zero. The graph
    is unchanged. Time is O(n + m), with O(n) extra space for n vertices and m
    edges; the empty graph returns an empty dictionary.
    """
    return _greedy_coloring(graph, graph.vertices)


def smallest_last_ordering(graph: Graph) -> list[int]:
    """Return the reverse of successive minimum-current-degree removals.

    Degree buckets support removals and neighbor updates in O(n + m) time and
    O(n) extra space. Bucket insertion order breaks ties deterministically.
    The graph is unchanged.
    """
    degrees = {vertex: graph.degree(vertex) for vertex in graph.vertices}
    # Dictionary buckets allow constant-time removal and retain insertion order.
    buckets = [{} for _ in range(max(degrees.values(), default=0) + 1)]
    for vertex, degree in degrees.items():
        buckets[degree][vertex] = None

    removed = []
    minimum = 0
    while degrees:
        while not buckets[minimum]:
            minimum += 1
        # popitem resolves equal-degree ties with the most recent bucket insertion.
        vertex, _ = buckets[minimum].popitem()
        del degrees[vertex]
        removed.append(vertex)
        for neighbor in graph.neighbors(vertex):
            if neighbor in degrees:
                degree = degrees[neighbor]
                del buckets[degree][neighbor]
                degrees[neighbor] = degree - 1
                buckets[degree - 1][neighbor] = None
                # A degree decrease may create a new bucket below the current minimum.
                minimum = min(minimum, degree - 1)
    # Reversing removals bounds already colored neighbors by the graph's degeneracy.
    return removed[::-1]


def smallest_last_coloring(graph: Graph) -> dict[int, int]:
    """Greedily color in smallest-last order, using at most degeneracy + 1 colors.

    Colors start at zero; an empty graph uses no colors. This heuristic does not
    guarantee an optimal coloring. Time is O(n + m), with O(n) extra space, and
    the graph is unchanged.
    """
    return _greedy_coloring(graph, smallest_last_ordering(graph))
