"""Run with: python -m run_benchmark --runs 30 --seed 0."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .benchmark import (
    benchmark_graph_coloring,
    benchmark_vertex_cover,
    read_results,
    write_results,
)


def plot_results(results: list[dict], output_dir: str | Path) -> None:
    """Write runtime, solution-size, excess, ratio, and randomized-STD plots.

    Pool runs and instances with the same vertex count (one supplied FRB
    family). Each runtime point uses that family's mean |V| + |E| and mean
    elapsed time. Only randomized vertex cover has an STD plot: calculate the
    sample STD across runs of each instance, then average those STDs per family.

    Args:
        results: Rows for one problem, with both algorithms at every vertex count
            and a shared optimum within each family.
        output_dir: Directory to create or overwrite PNG plots in. An obsolete
            combined plot named ``{problem}.png`` is removed if present.

    Raises:
        ValueError: If no benchmark results are supplied.
        OSError: If plot files cannot be created, written, or removed.
    """
    if not results:
        raise ValueError("No benchmark results to plot")
    coloring = "color_count" in results[0]
    size_metric = "color_count" if coloring else "cover_size"
    symbol = "k" if coloring else "C"
    problem = "graph_coloring" if coloring else "vertex_cover"
    title = "Graph coloring" if coloring else "Vertex cover"
    algorithms = (
        (("greedy", "Greedy"), ("smallest_last", "Smallest last"))
        if coloring
        else (("approx", "2-approximation"), ("randomized", "Randomized"))
    )
    groups = defaultdict(list)
    optima = {}
    for row in results:
        groups[row["algorithm"], row["vertices"]].append(row)
        optima[row["vertices"]] = row["optimum"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for algorithm, name in algorithms:
        points = sorted(
            (
                mean(row["vertices"] + row["edges"] for row in rows),
                mean(row["runtime_seconds"] for row in rows),
            )
            for (algo, _), rows in groups.items() if algo == algorithm
        )
        ax.plot(
            [size for size, _ in points],
            [runtime for _, runtime in points],
            marker="o",
            label=name,
        )
    ax.set_title(f"{title}: average runtime per family")
    ax.set_xlabel("Average graph size |V| + |E|")
    ax.set_ylabel("Average runtime (seconds)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / f"{problem}_runtime.png", dpi=160)
    plt.close(fig)

    sizes = sorted(optima)
    positions = list(range(len(sizes)))
    width = 0.23
    spacing = 0.34
    fig, ax = plt.subplots(
        figsize=(max(9, 1.5 * len(sizes)), 6), layout="constrained"
    )
    optimum_bars = ax.bar(
        [x - spacing for x in positions],
        [optima[n] for n in sizes],
        width=width,
        color="0.75",
        edgecolor="0.4",
        label="OPT",
    )
    ax.bar_label(optimum_bars, fmt="%g", padding=5, fontsize=8)
    upper = max(optima.values())
    for index, (algorithm, name) in enumerate(algorithms):
        measurements = [groups[algorithm, n] for n in sizes]
        averages = [mean(row[size_metric] for row in rows) for rows in measurements]
        bars = ax.bar(
            [x + index * spacing for x in positions],
            averages,
            width=width,
            color=f"C{index}",
            label=name,
        )
        ax.bar_label(bars, fmt="%.1f", padding=5, fontsize=8)
        upper = max(upper, *averages)
    ax.set_title(f"{title}: mean result versus OPT")
    ax.set_xlabel("Vertices |V|")
    ax.set_ylabel("Number of colors k" if coloring else "Vertex cover size C")
    ax.set_xticks(positions, [str(n) for n in sizes])
    ax.set_ylim(0, max(upper, 1) * 1.15)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    size_name = "color_count" if coloring else "size"
    fig.savefig(output_dir / f"{problem}_{size_name}.png", dpi=160)
    plt.close(fig)

    for metric, label, baseline in (
        ("excess", f"Mean excess ({symbol} − OPT)", 0),
        ("factor", f"Mean ratio ({symbol} / OPT)", 1),
    ):
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        for algorithm, name in algorithms:
            ax.plot(
                sizes,
                [mean(row[metric] for row in groups[algorithm, n]) for n in sizes],
                marker="o",
                label=name,
            )
        ax.axhline(baseline, color="black", linestyle="--", label="OPT")
        ax.set_title(f"{title}: {label}")
        ax.set_xlabel("Vertices |V|")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.savefig(output_dir / f"{problem}_{metric}.png", dpi=160)
        plt.close(fig)

    if not coloring:
        # Compute variation within each instance before averaging across a family.
        instances = defaultdict(list)
        for row in results:
            if row["algorithm"] == "randomized":
                instances[row["vertices"], row["instance"]].append(row["cover_size"])
        deviations = defaultdict(list)
        for (vertices, _), values in instances.items():
            deviations[vertices].append(stdev(values) if len(values) > 1 else 0.0)
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        ax.plot(
            sizes,
            [mean(deviations[n]) for n in sizes],
            color="C1",
            marker="o",
            label="Randomized",
        )
        ax.set_title("Randomized vertex cover: variation across repeated runs")
        ax.set_xlabel("Vertices |V|")
        ax.set_ylabel("Mean per-instance STD of cover size")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.savefig(output_dir / "vertex_cover_std.png", dpi=160)
        plt.close(fig)
    (output_dir / f"{problem}.png").unlink(missing_ok=True)


def main() -> None:
    """Parse CLI options and run benchmarks, write CSVs, or plot saved results.

    Search the selected data directories recursively and report progress to
    standard output. Plot mode reads raw CSVs from the output directory and
    requires no datasets. Invalid arguments, unavailable inputs, and invalid
    benchmark results are reported through argparse with exit status 2.
    """
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Benchmark vertex-cover and graph-coloring algorithms"
    )
    parser.add_argument(
        "--mode", choices=("all", "csv", "plot"), default="all",
        help="all: benchmark, CSVs, and plots; csv: benchmark and CSVs; "
        "plot: plots from saved CSVs (default: all)",
    )
    parser.add_argument(
        "--problem", choices=("all", "vertex-cover", "graph-coloring"), default="all",
        help="problem to benchmark (default: all)",
    )
    parser.add_argument(
        "--runs", type=int, default=30,
        help="measured runs per algorithm and instance (default: 30)",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="untimed warm-up runs per algorithm and instance; 0 disables "
        "warm-up (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="fixed graph-shuffle seed and first randomized endpoint seed (default: 0)",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        help="directory searched recursively for datasets (default: supplied data)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "result",
        help="CSV and plot directory; plot mode reads raw CSVs here (default: result/)",
    )
    args = parser.parse_args()
    problems = (
        ("vertex-cover", "graph-coloring") if args.problem == "all" else (args.problem,)
    )
    if args.mode == "plot":
        # Load and validate every requested CSV before replacing any plots.
        cached_results = []
        for problem in problems:
            path = args.output_dir / f"{problem.replace('-', '_')}_runs.csv"
            try:
                results = read_results(path)
                metric = "color_count" if problem == "graph-coloring" else "cover_size"
                if metric not in results[0]:
                    raise ValueError(f"Expected {metric} results in {path}")
                cached_results.append(results)
            except (OSError, ValueError) as error:
                parser.error(str(error))
        for results in cached_results:
            plot_results(results, args.output_dir)
        print(f"Saved plots from CSVs to {args.output_dir.resolve()}")
        return

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    # Resolve every requested dataset before starting work or writing outputs.
    datasets = []
    for problem in problems:
        coloring = problem == "graph-coloring"
        data_dir = args.data_dir if args.data_dir is not None else root / (
            "data/frb-gragh-coloring" if coloring else "data/frb-vertex-cover"
        )
        extension = ".clq" if coloring else ".mis"
        if not data_dir.is_dir():
            parser.error(f"Dataset directory does not exist: {data_dir}")
        paths = sorted(data_dir.rglob(f"*{extension}"))
        if not paths:
            parser.error(f"No {extension} instances found in {data_dir}")
        datasets.append((problem, paths))

    for problem, paths in datasets:
        results = []
        for path in paths:
            print(
                f"Benchmarking {path.name} ({args.warmup} warm-up runs and "
                f"{args.runs} measured runs per algorithm)",
                flush=True,
            )
            try:
                if problem == "graph-coloring":
                    results.extend(benchmark_graph_coloring(
                        path, args.runs, args.seed, warmup=args.warmup
                    ))
                else:
                    results.extend(benchmark_vertex_cover(
                        path, args.runs, args.seed, warmup=args.warmup
                    ))
            except (OSError, ValueError) as error:
                parser.error(str(error))
        write_results(results, args.output_dir)
        if args.mode == "all":
            plot_results(results, args.output_dir)
    outputs = "CSVs and plots" if args.mode == "all" else "CSVs"
    print(f"Saved {outputs} to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
