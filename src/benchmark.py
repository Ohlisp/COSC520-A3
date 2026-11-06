"""Repeated vertex-cover and coloring measurements on FRB DIMACS instances."""

import csv
import re
from collections import defaultdict
from math import isfinite
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

from .graph import Graph
from .graph_coloring import greedy_coloring, smallest_last_coloring
from .vertex_cover import approx_vertex_cover, randomized_vertex_cover


def benchmark_vertex_cover(
    path: str | Path, runs: int = 30, seed: int = 0, *, warmup: int = 5
) -> list[dict]:
    """Measure both algorithms, loading the graph once outside the timer.

    For an FRB instance named frbK-S-I.mis, |V| = K*S and the known maximum
    independent set has size K, so OPT = |V| - K. Shuffle vertices and edges
    once using seed, then reuse that graph for both algorithms and every run.
    Run warmup untimed calls per algorithm before collecting measurements.
    Randomized endpoint choices use seed+i for run i (starting at zero), with
    that sequence restarted after warm-up. Only measured algorithm calls are
    timed; shuffling, warm-up, validation, and metrics are outside the timer.

    Args:
        path: DIMACS file whose FRB filename supplies the known optimum.
        runs: Positive number of measured calls per algorithm.
        seed: Fixed shuffle seed and first randomized endpoint-choice seed.
        warmup: Nonnegative number of untimed calls per algorithm.

    Returns:
        Two rows per measured run, containing instance metadata, seeds, cover
        size, runtime in seconds, excess above OPT, and ratio to OPT.

    Raises:
        OSError: If the dataset cannot be opened or read.
        ValueError: If counts, dataset metadata, or a returned cover are invalid.
    """
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    path = Path(path)
    match = re.fullmatch(r"frb([1-9]\d*)-([1-9]\d*)-[1-9]\d*\.mis", path.name)
    if match is None:
        raise ValueError(f"Expected an FRB filename frbK-S-I.mis: {path.name}")
    cliques, clique_size = map(int, match.groups())
    graph = Graph.from_dimacs(path)
    if len(graph) != cliques * clique_size:
        raise ValueError(f"Vertex count does not match FRB filename: {path.name}")
    optimum = len(graph) - cliques
    if optimum <= 0:
        raise ValueError("FRB optimum must be positive to calculate C/OPT")

    vertices = set(graph.vertices)
    shuffled = graph.shuffled(seed)
    for run in range(warmup):
        approx_vertex_cover(shuffled)
        randomized_vertex_cover(shuffled, seed=seed + run)

    results = []
    for run in range(runs):
        # Restart the seed sequence so warm-up cannot change measured solutions.
        run_seed = seed + run
        for name, algorithm in (
            ("approx", approx_vertex_cover),
            ("randomized", randomized_vertex_cover),
        ):
            kwargs = {"seed": run_seed} if name == "randomized" else {}
            start = perf_counter()
            cover = algorithm(shuffled, **kwargs)
            elapsed = perf_counter() - start

            # Validate against the original graph outside the measured interval.
            if not cover <= vertices or any(
                u not in cover and v not in cover for u, v in graph.edges
            ):
                raise ValueError(f"{name} returned an invalid cover for {path.name}")
            size = len(cover)
            results.append(
                {
                    "instance": path.stem,
                    "vertices": len(graph),
                    "edges": len(graph.edges),
                    "optimum": optimum,
                    "algorithm": name,
                    "run": run + 1,
                    "shuffle_seed": seed,
                    "seed": run_seed if name == "randomized" else None,
                    "cover_size": size,
                    "runtime_seconds": elapsed,
                    "excess": size - optimum,
                    "factor": size / optimum,
                }
            )
            del cover
    return results


def benchmark_graph_coloring(
    path: str | Path, runs: int = 30, seed: int = 0, *, warmup: int = 5
) -> list[dict]:
    """Measure both colorings against the known FRB chromatic number K.

    An instance named frbK-S-I.clq has K*S vertices and OPT = K. Load the graph
    once; shuffle vertices and edges once using seed, then reuse that graph for
    both algorithms and every run. Repetition measures runtime variation; both
    algorithms return the same color count for the fixed input. Run warmup
    untimed calls per algorithm before collecting measurements. Time only
    measured algorithm calls, then validate colors and calculate metrics.

    Args:
        path: DIMACS file whose FRB filename supplies the known chromatic number.
        runs: Positive number of measured calls per algorithm.
        seed: Fixed seed for the once-per-instance graph shuffle.
        warmup: Nonnegative number of untimed calls per algorithm.

    Returns:
        Two rows per measured run, containing instance metadata, shuffle seed,
        distinct color count, runtime in seconds, excess, and ratio to OPT.

    Raises:
        OSError: If the dataset cannot be opened or read.
        ValueError: If counts, dataset metadata, or a returned coloring are invalid.
    """
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    path = Path(path)
    match = re.fullmatch(r"frb([1-9]\d*)-([1-9]\d*)-[1-9]\d*\.clq", path.name)
    if match is None:
        raise ValueError(f"Expected an FRB filename frbK-S-I.clq: {path.name}")
    optimum, clique_size = map(int, match.groups())
    graph = Graph.from_dimacs(path)
    if len(graph) != optimum * clique_size:
        raise ValueError(f"Vertex count does not match FRB filename: {path.name}")

    vertices = set(graph.vertices)
    shuffled = graph.shuffled(seed)
    for _ in range(warmup):
        greedy_coloring(shuffled)
        smallest_last_coloring(shuffled)

    results = []
    for run in range(runs):
        for name, algorithm in (
            ("greedy", greedy_coloring),
            ("smallest_last", smallest_last_coloring),
        ):
            start = perf_counter()
            colors = algorithm(shuffled)
            elapsed = perf_counter() - start

            if (
                colors.keys() != vertices
                or any(not isinstance(color, int) or color < 0 for color in colors.values())
                or any(colors[u] == colors[v] for u, v in graph.edges)
            ):
                raise ValueError(f"{name} returned an invalid coloring for {path.name}")
            # Color labels need not be consecutive, so count distinct values.
            count = len(set(colors.values()))
            results.append(
                {
                    "instance": path.stem,
                    "vertices": len(graph),
                    "edges": len(graph.edges),
                    "optimum": optimum,
                    "algorithm": name,
                    "run": run + 1,
                    "shuffle_seed": seed,
                    "seed": None,
                    "color_count": count,
                    "runtime_seconds": elapsed,
                    "excess": count - optimum,
                    "factor": count / optimum,
                }
            )
            del colors
    return results


def summarize_results(results: list[dict]) -> list[dict]:
    """Group raw runs by instance and algorithm and summarize their metrics.

    ``results`` contains rows from one benchmark problem. Return metadata and
    the mean and sample standard deviation of solution size, runtime, excess,
    and ratio for each group, in first-seen order. A single run has standard
    deviation zero by convention; empty input returns an empty list.
    """
    groups = defaultdict(list)
    for row in results:
        groups[row["instance"], row["algorithm"]].append(row)

    summary = []
    for group in groups.values():
        row = {
            key: group[0][key]
            for key in ("instance", "vertices", "edges", "optimum", "algorithm")
        }
        if "shuffle_seed" in group[0]:
            row["shuffle_seed"] = group[0]["shuffle_seed"]
        row["runs"] = len(group)
        size_metric = "color_count" if "color_count" in group[0] else "cover_size"
        for metric in (size_metric, "runtime_seconds", "excess", "factor"):
            values = [run[metric] for run in group]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary.append(row)
    return summary


def write_results(results: list[dict], output_dir: str | Path) -> None:
    """Write raw runs and per-instance summaries as UTF-8 CSV files.

    ``results`` must contain rows from one benchmark problem. Create
    ``output_dir`` as needed and overwrite its ``{problem}_runs.csv`` and
    ``{problem}_summary.csv`` files, where the problem is inferred from the
    solution-size column (``cover_size`` or ``color_count``).

    Raises:
        ValueError: If no results are supplied.
        OSError: If the output directory or files cannot be created or written.
    """
    if not results:
        raise ValueError("No benchmark results to write")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problem = "graph_coloring" if "color_count" in results[0] else "vertex_cover"
    for filename, rows in (
        (f"{problem}_runs.csv", results),
        (f"{problem}_summary.csv", summarize_results(results)),
    ):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def read_results(path: str | Path) -> list[dict]:
    """Load raw CSV runs, converting and validating the fields used by plots.

    Convert counts, run numbers, and shuffle seeds to integers and runtime,
    excess, and ratio to floats; leave other columns as strings. Require both
    algorithms for each vertex count, one shuffle seed per instance, and fixed
    solution sizes for deterministic algorithms.

    Args:
        path: Raw runs CSV produced by ``write_results``.

    Returns:
        Nonempty benchmark rows in CSV order, ready for plotting.

    Raises:
        OSError: If the CSV cannot be opened or read.
        ValueError: If columns, values, or benchmark consistency checks fail.
    """
    path = Path(path)
    results = []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or ())
        size_fields = fields & {"cover_size", "color_count"}
        if len(size_fields) != 1:
            raise ValueError(f"Expected cover_size or color_count column in {path}")
        size_metric = size_fields.pop()
        if "shuffle_seed" not in fields:
            raise ValueError(
                f"Missing shuffle_seed in {path}; regenerate CSVs with --mode csv "
                "to shuffle each instance only once"
            )
        integers = ("vertices", "edges", "optimum", size_metric, "run", "shuffle_seed")
        decimals = ("runtime_seconds", "excess", "factor")
        missing = set((*integers, *decimals, "algorithm", "instance")) - fields
        if missing:
            raise ValueError(f"Missing CSV columns in {path}: {', '.join(sorted(missing))}")
        algorithms = (
            {"greedy", "smallest_last"} if size_metric == "color_count"
            else {"approx", "randomized"}
        )
        groups = defaultdict(set)
        shuffle_seeds = defaultdict(set)
        deterministic_sizes = defaultdict(set)
        for line, row in enumerate(reader, start=2):
            try:
                for key in integers:
                    row[key] = int(row[key])
                for key in decimals:
                    row[key] = float(row[key])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid numeric value in {path} on line {line}"
                ) from error
            if (
                row["algorithm"] not in algorithms
                or not row["instance"]
                or any(row[key] < 0 for key in integers if key != "shuffle_seed")
                or row["run"] == 0
                or row["optimum"] == 0
                or row["runtime_seconds"] < 0
                or not all(isfinite(row[key]) for key in decimals)
            ):
                raise ValueError(f"Invalid benchmark result in {path} on line {line}")
            groups[row["vertices"]].add(row["algorithm"])
            shuffle_seeds[row["instance"]].add(row["shuffle_seed"])
            if row["algorithm"] != "randomized":
                deterministic_sizes[row["instance"], row["algorithm"]].add(
                    row[size_metric]
                )
            results.append(row)
    if not results:
        raise ValueError(f"No benchmark results in {path}")
    if any(group != algorithms for group in groups.values()):
        raise ValueError(f"Expected both algorithms for every vertex count in {path}")
    # Fixed inputs make deterministic solution sizes constant across repeated runs.
    if any(len(seeds) != 1 for seeds in shuffle_seeds.values()):
        raise ValueError(f"Expected one fixed shuffle_seed per instance in {path}")
    if any(len(sizes) != 1 for sizes in deterministic_sizes.values()):
        raise ValueError(
            f"Deterministic solution sizes vary across runs in {path}; regenerate CSVs"
        )
    return results
