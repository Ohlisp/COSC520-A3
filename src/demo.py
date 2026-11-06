"""Run ``python -m src.demo`` for a small, validated demo with text diagrams."""

from .graph import Graph
from .graph_coloring import greedy_coloring, smallest_last_coloring
from .vertex_cover import approx_vertex_cover, randomized_vertex_cover


def _print_graph(labels: dict[int, str]) -> None:
    """Print the cube's two four-cycles and four connecting edges as text.

    ``labels`` annotates vertices 1 through 8. The outer lines connect vertices
    1 to 4 and 5 to 8; the four middle vertical lines connect the two rows.
    """
    print("    +-----------------------+")
    print("    |                       |")
    print("  " + "---".join(f"[{labels[v]:^3}]" for v in (1, 2, 3, 4)))
    print("    |       |       |       |")
    print("  " + "---".join(f"[{labels[v]:^3}]" for v in (5, 6, 7, 8)))
    print("    |                       |")
    print("    +-----------------------+")


def main() -> None:
    """Validate both covers and colorings and print their annotated cube graphs.

    A star marks each selected cover vertex; coloring labels show vertex:color.
    Raise ValueError if a cover misses an edge, a coloring is invalid, or the
    input graph changes.
    """
    # Every vertex has degree three. These nonadjacent pairs make greedy use four colors.
    graph = Graph(
        [
            (1, 2), (2, 3), (3, 4), (4, 1),
            (5, 6), (6, 7), (7, 8), (8, 5),
            (1, 5), (2, 6), (3, 7), (4, 8),
        ],
        vertices=[1, 7, 3, 5, 6, 4, 8, 2],
    )
    before = (graph.vertices, graph.edges, [graph.neighbors(v) for v in graph.vertices])
    covers = [
        ("2-approximation cover", approx_vertex_cover(graph)),
        ("Randomized cover (seed 0)", randomized_vertex_cover(graph, seed=0)),
    ]
    colorings = [
        ("Greedy coloring", greedy_coloring(graph)),
        ("Smallest-last coloring", smallest_last_coloring(graph)),
    ]
    print(f"Cube graph: {len(graph)} vertices, {len(graph.edges)} edges")
    print("Vertex degrees:", {v: graph.degree(v) for v in sorted(graph.vertices)})
    _print_graph({v: str(v) for v in graph.vertices})
    print(f"Greedy input order: {graph.vertices}")
    print("\nVertex cover (* = selected):")
    for name, cover in covers:
        if not cover <= set(graph.vertices) or any(
            u not in cover and v not in cover for u, v in graph.edges
        ):
            raise ValueError(f"{name} returned an invalid cover")
        print(f"{name}: {len(cover)} selected; valid, all {len(graph.edges)} edges covered")
        _print_graph({v: f"{v}*" if v in cover else str(v) for v in graph.vertices})

    print("\nGraph coloring (vertex:color):")
    for name, colors in colorings:
        if (
            colors.keys() != set(graph.vertices)
            or any(not isinstance(c, int) or c < 0 for c in colors.values())
            or any(colors[u] == colors[v] for u, v in graph.edges)
        ):
            raise ValueError(f"{name} returned an invalid coloring")
        print(f"{name}: {len(set(colors.values()))} colors; valid, no edge conflicts")
        _print_graph({v: f"{v}:{colors[v]}" for v in graph.vertices})
    after = (graph.vertices, graph.edges, [graph.neighbors(v) for v in graph.vertices])
    if after != before:
        raise ValueError("An algorithm changed the graph")
    print("\nInput graph unchanged: valid")


if __name__ == "__main__":
    main()
