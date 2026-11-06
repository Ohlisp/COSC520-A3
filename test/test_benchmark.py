"""Check benchmark metrics, reproducibility, and generated artifacts."""

import csv
import subprocess
import sys
from math import sqrt
from pathlib import Path

import pytest

from src import benchmark as benchmark_module
from src.benchmark import benchmark_vertex_cover, summarize_results, write_results
from src.graph import Graph


@pytest.fixture
def instance(tmp_path):
    """Create a DIMACS path whose minimum cover matches its FRB filename."""
    # A four-vertex path has MIS = 2 and MVC = 2, matching frb2-2 metadata.
    path = tmp_path / "frb2-2-1.mis"
    path.write_text("p edge 4 3\ne 1 2\ne 2 3\ne 3 4\n", encoding="utf-8")
    return path


def test_run_metrics_and_sample_statistics(instance, monkeypatch):
    """Check run metadata, quality metrics, and sample statistics with fixed timing."""
    seeds = []

    def randomized(graph, seed):
        """Record each endpoint seed and return a cover with a known size."""
        seeds.append(seed)
        return {1, 2, 3} if seed == 8 else {2, 3}

    monkeypatch.setattr("src.benchmark.randomized_vertex_cover", randomized)
    monkeypatch.setattr(Graph, "shuffled", lambda self, seed: self)
    durations = [0.25, 0.25, 0.5, 0.5, 0.75, 0.75]
    ticks = iter(
        tick
        for start, elapsed in enumerate(durations)
        for tick in (start, start + elapsed)
    )
    monkeypatch.setattr("src.benchmark.perf_counter", lambda: next(ticks))

    rows = benchmark_vertex_cover(instance, runs=3, seed=7, warmup=0)

    assert len(rows) == 6
    assert seeds == [7, 8, 9]
    assert [row["runtime_seconds"] for row in rows] == durations
    assert all(row["vertices"] == 4 and row["edges"] == 3 for row in rows)
    assert all(row["optimum"] == 2 for row in rows)
    assert [row["cover_size"] for row in rows] == [4, 2, 4, 3, 4, 2]
    assert [row["excess"] for row in rows] == [2, 0, 2, 1, 2, 0]
    assert [row["factor"] for row in rows] == [2, 1, 2, 1.5, 2, 1]
    assert [row["run"] for row in rows] == [1, 1, 2, 2, 3, 3]
    assert [row["seed"] for row in rows] == [None, 7, None, 8, None, 9]
    assert [row["shuffle_seed"] for row in rows] == [7] * 6

    deterministic, randomized_summary = summarize_results(rows)
    assert deterministic["cover_size_mean"] == 4
    assert deterministic["cover_size_std"] == 0
    assert deterministic["runtime_seconds_mean"] == 0.5
    assert deterministic["runtime_seconds_std"] == 0.25
    assert randomized_summary["runs"] == 3
    assert randomized_summary["cover_size_mean"] == pytest.approx(7 / 3)
    assert randomized_summary["cover_size_std"] == pytest.approx(1 / sqrt(3))
    assert randomized_summary["excess_mean"] == pytest.approx(1 / 3)
    assert randomized_summary["factor_mean"] == pytest.approx(7 / 6)


def test_seeded_cover_sizes_are_reproducible(instance):
    """Check measured cover sizes repeat even when the warm-up count changes."""
    # A cycle permits different endpoint choices to change cover size even
    # while its input order stays fixed; its MVC still matches frb2-2 metadata.
    instance.write_text(
        "p edge 4 4\ne 1 2\ne 2 3\ne 3 4\ne 4 1\n", encoding="utf-8"
    )
    first = benchmark_vertex_cover(instance, runs=30, seed=42, warmup=0)
    second = benchmark_vertex_cover(instance, runs=30, seed=42, warmup=7)

    assert [row["cover_size"] for row in first] == [row["cover_size"] for row in second]
    randomized_sizes = {
        row["cover_size"] for row in first if row["algorithm"] == "randomized"
    }
    assert len(randomized_sizes) > 1
    assert len({row["cover_size"] for row in first if row["algorithm"] == "approx"}) == 1


@pytest.mark.parametrize("warmup", [0, 3, None])
@pytest.mark.parametrize(
    "extension, function_name, algorithms",
    [
        (
            ".mis", "benchmark_vertex_cover",
            ("approx_vertex_cover", "randomized_vertex_cover"),
        ),
        (
            ".clq", "benchmark_graph_coloring",
            ("greedy_coloring", "smallest_last_coloring"),
        ),
    ],
)
def test_benchmarks_warm_up_shared_inputs_before_timing(
    instance, monkeypatch, extension, function_name, algorithms, warmup
):
    """Verify warm-up stays untimed and all calls share one seeded graph shuffle."""
    path = instance.with_suffix(extension)
    if path != instance:
        path.write_bytes(instance.read_bytes())
    original = Graph.from_dimacs(path)
    shuffle = Graph.shuffled
    seeds, seen = [], []
    timing = False
    ticks = iter(range(12))

    def clock():
        """Toggle the timing marker and supply successive synthetic clock ticks."""
        nonlocal timing
        timing = not timing
        return next(ticks)

    def shuffled(graph, seed):
        """Record shuffle seeds and assert shuffling occurs outside the timer."""
        assert not timing
        seeds.append(seed)
        return shuffle(graph, seed)

    def observe(algorithm):
        """Wrap an algorithm to record its graph, timing state, and endpoint seed."""
        def measured(graph, **kwargs):
            """Record call context before forwarding to the original algorithm."""
            seen.append((graph, timing, kwargs.get("seed")))
            return algorithm(graph, **kwargs)

        return measured

    monkeypatch.setattr(Graph, "shuffled", shuffled)
    monkeypatch.setattr(benchmark_module, "perf_counter", clock)
    for name in algorithms:
        monkeypatch.setattr(
            benchmark_module, name, observe(getattr(benchmark_module, name))
        )

    options = {} if warmup is None else {"warmup": warmup}
    rows = getattr(benchmark_module, function_name)(path, runs=3, seed=11, **options)
    warmup_count = 5 if warmup is None else warmup

    assert seeds == [11]
    assert len(rows) == 6
    expected_seeds = [None, 11, None, 12, None, 13] if extension == ".mis" else [None] * 6
    assert [row["seed"] for row in rows] == expected_seeds
    assert [row["shuffle_seed"] for row in rows] == [11] * 6
    assert [row["run"] for row in rows] == [1, 1, 2, 2, 3, 3]
    assert all(row["runtime_seconds"] == 1 for row in rows)
    graphs, measured, endpoint_seeds = zip(*seen)
    assert list(measured) == [False] * (2 * warmup_count) + [True] * 6
    warmup_seeds = (
        [value for run in range(warmup_count) for value in (None, 11 + run)]
        if extension == ".mis" else [None] * (2 * warmup_count)
    )
    assert list(endpoint_seeds) == warmup_seeds + expected_seeds
    for first, second in zip(graphs[::2], graphs[1::2]):
        assert first is second
        assert set(first.vertices) == set(original.vertices)
        assert set(first.edges) == set(original.edges)
    assert len({id(graph) for graph in graphs}) == 1
    assert not timing


def test_single_run_has_zero_standard_deviation(instance, tmp_path):
    """Check single-run CSV summaries use zero for every standard deviation."""
    rows = benchmark_vertex_cover(instance, runs=1)
    output = tmp_path / "result"

    write_results(rows, output)

    with (output / "vertex_cover_summary.csv").open(newline="") as source:
        summary = list(csv.DictReader(source))
    assert len(summary) == 2
    for row in summary:
        assert row["runs"] == "1"
        assert all(
            float(value) == 0 for key, value in row.items() if key.endswith("_std")
        )


@pytest.mark.parametrize("runs", [0, -1])
def test_nonpositive_run_counts_are_rejected(instance, runs):
    """Require at least one measured run for vertex-cover benchmarks."""
    with pytest.raises(ValueError, match="runs must be at least 1"):
        benchmark_vertex_cover(instance, runs=runs)


@pytest.mark.parametrize(
    "benchmark",
    [benchmark_module.benchmark_vertex_cover, benchmark_module.benchmark_graph_coloring],
)
def test_negative_warmup_counts_are_rejected(instance, benchmark):
    """Require nonnegative warm-up counts for both benchmark problems."""
    with pytest.raises(ValueError, match="warmup must be nonnegative"):
        benchmark(instance, warmup=-1)


@pytest.mark.parametrize(
    "filename, contents, message",
    [
        ("unknown.mis", "p edge 4 0\n", "Expected an FRB filename"),
        ("frb3-2-1.mis", "p edge 4 0\n", "Vertex count does not match"),
        ("frb4-1-1.mis", "p edge 4 0\n", "optimum must be positive"),
    ],
)
def test_invalid_metadata_is_rejected(tmp_path, filename, contents, message):
    """Reject invalid FRB names, mismatched vertex counts, and nonpositive OPT."""
    path = tmp_path / filename
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        benchmark_vertex_cover(path)


@pytest.mark.parametrize("cover", [set(), {1, 2, 3, 99}])
def test_invalid_covers_are_rejected(instance, monkeypatch, cover):
    """Reject algorithm results that miss edges or contain unknown vertices."""
    monkeypatch.setattr("src.benchmark.approx_vertex_cover", lambda graph: cover)

    with pytest.raises(ValueError, match="invalid cover"):
        benchmark_vertex_cover(instance, runs=1)


def test_empty_results_are_rejected(tmp_path):
    """Require at least one benchmark row before writing CSV files."""
    with pytest.raises(ValueError, match="No benchmark results"):
        write_results([], tmp_path)


def test_cli_finds_nested_instances_and_writes_csvs_and_plot(instance, tmp_path):
    """Check recursive cover dataset discovery and generated CSV and PNG files."""
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "frb2-2-2.mis").write_bytes(instance.read_bytes())
    output = tmp_path / "result"

    completed = subprocess.run(
        [
            sys.executable,
            "-m", "src.run_benchmark",
            "--problem", "vertex-cover",
            "--data-dir", str(tmp_path),
            "--output-dir", str(output),
            "--runs", "2",
            "--seed", "7",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Saved CSVs and plot" in completed.stdout
    with (output / "vertex_cover_runs.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    with (output / "vertex_cover_summary.csv").open(newline="") as source:
        summary = list(csv.DictReader(source))
    assert len(rows) == 8
    assert len(summary) == 4
    assert {row["instance"] for row in rows} == {"frb2-2-1", "frb2-2-2"}
    assert {row["seed"] for row in rows if row["algorithm"] == "randomized"} == {
        "7", "8"
    }
    assert all(row["optimum"] == "2" for row in rows)
    for suffix in ("runtime", "size", "excess", "factor", "std"):
        filename = f"vertex_cover_{suffix}.png"
        assert (output / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize(
    "case", ["invalid-runs", "invalid-warmup", "missing-directory", "empty-directory"]
)
def test_cli_reports_input_errors(tmp_path, case):
    """Check CLI errors for invalid run counts, warm-up counts, and dataset paths."""
    data_dir = tmp_path / "missing" if case == "missing-directory" else tmp_path
    completed = subprocess.run(
        [
            sys.executable,
            "-m", "src.run_benchmark",
            "--data-dir", str(data_dir),
            "--runs", "0" if case == "invalid-runs" else "1",
            "--warmup", "-1" if case == "invalid-warmup" else "5",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert {
        "invalid-runs": "--runs must be at least 1",
        "invalid-warmup": "--warmup must be nonnegative",
        "missing-directory": "Dataset directory does not exist",
        "empty-directory": "No .mis instances found",
    }[case] in completed.stderr


@pytest.mark.parametrize("problem_args", [[], ["--problem", "all"]])
def test_cli_runs_both_problems_and_writes_all_outputs(instance, tmp_path, problem_args):
    """Check the default and explicit all-problem modes produce every artifact."""
    nested = tmp_path / "coloring" / "nested"
    nested.mkdir(parents=True)
    (nested / "frb2-2-1.clq").write_bytes(instance.read_bytes())
    output = tmp_path / "result"

    completed = subprocess.run(
        [
            sys.executable,
            "-m", "run_benchmark",
            *problem_args,
            "--data-dir", str(tmp_path),
            "--output-dir", str(output),
            "--runs", "2",
            "--seed", "7",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Saved CSVs and plots" in completed.stdout
    assert {path.name for path in output.iterdir()} == {
        "vertex_cover_runs.csv", "vertex_cover_summary.csv",
        "vertex_cover_runtime.png", "vertex_cover_size.png",
        "vertex_cover_excess.png", "vertex_cover_factor.png", "vertex_cover_std.png",
        "graph_coloring_runs.csv", "graph_coloring_summary.csv",
        "graph_coloring_runtime.png", "graph_coloring_color_count.png",
        "graph_coloring_excess.png", "graph_coloring_factor.png",
    }
    for problem, algorithms, metric in (
        ("vertex_cover", {"approx", "randomized"}, "cover_size"),
        ("graph_coloring", {"greedy", "smallest_last"}, "color_count"),
    ):
        with (output / f"{problem}_runs.csv").open(newline="") as source:
            rows = list(csv.DictReader(source))
        with (output / f"{problem}_summary.csv").open(newline="") as source:
            summary = list(csv.DictReader(source))
        assert len(rows) == 4
        assert len(summary) == 2
        assert {row["algorithm"] for row in rows} == algorithms
        assert all(row["optimum"] == "2" for row in rows)
        assert all(metric in row for row in rows)
        assert all(row["runs"] == "2" for row in summary)
        expected_seeds = ["", "7", "", "8"] if problem == "vertex_cover" else [""] * 4
        assert [row["seed"] for row in rows] == expected_seeds
        assert {row["shuffle_seed"] for row in rows} == {"7"}
        if problem == "vertex_cover":
            assert [row["seed"] for row in rows if row["algorithm"] == "randomized"] == [
                "7", "8"
            ]
        size_name = "color_count" if problem == "graph_coloring" else "size"
        suffixes = ["runtime", size_name, "excess", "factor"]
        if problem == "vertex_cover":
            suffixes.append("std")
        for suffix in suffixes:
            assert (output / f"{problem}_{suffix}.png").read_bytes().startswith(
                b"\x89PNG\r\n\x1a\n"
            )


def test_cli_checks_both_datasets_before_writing_outputs(instance, tmp_path):
    """Ensure a missing second problem stops the CLI before creating outputs."""
    output = tmp_path / "result"
    completed = subprocess.run(
        [
            sys.executable,
            "-m", "run_benchmark",
            "--data-dir", str(tmp_path),
            "--output-dir", str(output),
            "--runs", "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "No .clq instances found" in completed.stderr
    assert not output.exists()
