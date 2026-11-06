"""A simple undirected graph and a reader for the supplied DIMACS datasets."""

from collections.abc import Iterable
from copy import copy
from pathlib import Path
from random import Random


class Graph:
    """Store integer vertices, unique edges, and adjacency sets.

    Edge endpoints are added automatically; ``vertices`` can supply isolated
    vertices. Input order is preserved, duplicate edges are ignored, and
    self-loops are rejected. Vertices, edges, and neighbor sets are read-only.
    """

    def __init__(
        self,
        edges: Iterable[tuple[int, int]] = (),
        vertices: Iterable[int] = (),
    ) -> None:
        """Build an undirected graph from edges and optional isolated vertices.

        Args:
            edges: Endpoint pairs; repeated or reversed edges are stored once.
            vertices: Labels to insert before edge endpoints, including isolates.

        Raises:
            ValueError: If an edge connects a vertex to itself.
        """
        adjacency: dict[int, set[int]] = {vertex: set() for vertex in vertices}
        unique_edges = []
        for u, v in edges:
            if u == v:
                raise ValueError("Self-loops are not supported")
            adjacency.setdefault(u, set())
            adjacency.setdefault(v, set())
            # Symmetric adjacency detects duplicates in either orientation.
            if v not in adjacency[u]:
                adjacency[u].add(v)
                adjacency[v].add(u)
                unique_edges.append((u, v))

        # Freeze neighbor sets so callers and shuffled copies can safely share them.
        self._adjacency = {
            vertex: frozenset(neighbors) for vertex, neighbors in adjacency.items()
        }
        self._vertices = tuple(adjacency)
        self._edges = tuple(unique_edges)

    @property
    def vertices(self) -> tuple[int, ...]:
        """Return all vertex labels as an immutable tuple in stored order."""
        return self._vertices

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """Return unique edges in stored order, retaining their input orientation."""
        return self._edges

    def __len__(self) -> int:
        """Return the number of vertices, including isolated vertices."""
        return len(self._vertices)

    def neighbors(self, vertex: int) -> frozenset[int]:
        """Return a vertex's neighbors; raise KeyError for an unknown vertex."""
        return self._adjacency[vertex]

    def degree(self, vertex: int) -> int:
        """Return the neighbor count; raise KeyError for an unknown vertex."""
        return len(self.neighbors(vertex))

    def shuffled(self, seed: int | None = None) -> "Graph":
        """Return a copy with shuffled vertex and edge orders.

        Preserve labels and adjacency, sharing the read-only adjacency data.
        A local random generator makes seeded shuffles reproducible without
        changing global random state. The original graph is unchanged.

        Args:
            seed: Random seed, or None to seed from the generator's default source.

        Returns:
            A new graph with independently shuffled vertex and edge tuples.
        """
        rng = Random(seed)
        vertices = list(self.vertices)
        edges = list(self.edges)
        rng.shuffle(vertices)
        rng.shuffle(edges)
        graph = copy(self)
        graph._vertices = tuple(vertices)
        graph._edges = tuple(edges)
        return graph

    @classmethod
    def from_dimacs(cls, path: str | Path) -> "Graph":
        """Read ``p edge n m`` / ``e u v`` records, preserving labels 1 through n.

        Blank lines and ``c`` comments are ignored. Invalid records, duplicate
        edges, or counts inconsistent with the header raise ValueError.

        Args:
            path: Path to a UTF-8 DIMACS graph file.

        Returns:
            A graph containing every declared vertex, including isolates.

        Raises:
            OSError: If the input file cannot be opened or read.
            ValueError: If records, vertex labels, or declared counts are invalid.
        """
        counts = None
        edges = []
        with open(path, encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                fields = line.split()
                if not fields or fields[0] == "c":
                    continue
                if fields[0] == "p":
                    if counts is not None or len(fields) != 4 or fields[1] != "edge":
                        raise ValueError(f"Invalid DIMACS header on line {line_number}")
                    counts = int(fields[2]), int(fields[3])
                    if min(counts) < 0:
                        raise ValueError("Vertex and edge counts must be nonnegative")
                elif fields[0] == "e":
                    if counts is None or len(fields) != 3:
                        raise ValueError(f"Invalid DIMACS edge on line {line_number}")
                    u, v = int(fields[1]), int(fields[2])
                    if not (1 <= u <= counts[0] and 1 <= v <= counts[0]):
                        raise ValueError(f"Vertex out of range on line {line_number}")
                    edges.append((u, v))
                else:
                    raise ValueError(f"Unknown DIMACS record on line {line_number}")

        if counts is None:
            raise ValueError("Missing DIMACS header")
        if len(edges) != counts[1]:
            raise ValueError(f"Expected {counts[1]} edges, found {len(edges)}")
        # The constructor removes duplicates; DIMACS input must instead reject them.
        graph = cls(edges, vertices=range(1, counts[0] + 1))
        if len(graph.edges) != len(edges):
            raise ValueError("Duplicate edges in DIMACS input")
        return graph
