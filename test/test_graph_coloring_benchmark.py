"""Check coloring metrics, validation, and the benchmark command."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from src.benchmark import benchmark_graph_coloring, summarize_results, write_results


@pytest.fixture
def instance(tmp_path):
    """Create a two-colorable DIMACS path with an awkward greedy vertex order."""
    # A path with chromatic number two; input-order greedy uses three colors.
    path = tmp_path / "frb2-2-1.clq"
    path.write_text("p edge 4 3\ne 1 3\ne 3 4\ne 4 2\n", encoding="utf-8")
    return path


def test_run_metrics_and_sample_statistics(instance, monkeypatch):
    """Check coloring metrics and sample runtime statistics using a fixed clock."""
    monkeypatch.setattr("src.graph.Graph.shuffled", lambda self, seed: self)
    durations = [0.25, 0.25, 0.5, 0.5, 0.75, 0.75]
    ticks = iter(
        tick
        for start, elapsed in enumerate(durations)
        for tick in (start, start + elapsed)
    )
    monkeypatch.setattr("src.benchmark.perf_counter", lambda: next(ticks))

    rows = benchmark_graph_coloring(instance, runs=3, seed=7)

    assert len(rows) == 6
    assert [row["runtime_seconds"] for row in rows] == durations
    assert all(row["vertices"] == 4 and row["edges"] == 3 for row in rows)
    assert all(row["optimum"] == 2 for row in rows)
    assert [row["algorithm"] for row in rows] == ["greedy", "smallest_last"] * 3
    assert [row["color_count"] for row in rows] == [3, 2, 3, 2, 3, 2]
    assert [row["excess"] for row in rows] == [1, 0, 1, 0, 1, 0]
    assert [row["factor"] for row in rows] == [1.5, 1, 1.5, 1, 1.5, 1]
    assert [row["run"] for row in rows] == [1, 1, 2, 2, 3, 3]
    assert [row["seed"] for row in rows] == [None] * 6
    assert [row["shuffle_seed"] for row in rows] == [7] * 6

    greedy, smallest_last = summarize_results(rows)
    assert greedy["color_count_mean"] == 3
    assert greedy["color_count_std"] == 0
    assert greedy["runtime_seconds_mean"] == 0.5
    assert greedy["runtime_seconds_std"] == 0.25
    assert greedy["factor_mean"] == 1.5
    assert smallest_last["color_count_mean"] == 2
    assert smallest_last["excess_mean"] == 0
    assert smallest_last["factor_mean"] == 1


def test_fixed_shuffled_input_produces_repeatable_color_counts(instance):
    """Check deterministic color counts repeat for one fixed shuffled input."""
    first = benchmark_graph_coloring(instance, runs=16, seed=0)
    repeated = benchmark_graph_coloring(instance, runs=16, seed=0)

    assert [row["color_count"] for row in first] == [row["color_count"] for row in repeated]
    for algorithm in ("greedy", "smallest_last"):
        assert len({row["color_count"] for row in first if row["algorithm"] == algorithm}) == 1
    assert all(row["color_count_std"] == 0 for row in summarize_results(first))


def test_single_run_csvs_have_zero_standard_deviation(instance, tmp_path):
    """Verify raw coloring CSV metadata and zero single-run standard deviations."""
    rows = benchmark_graph_coloring(instance, runs=1)
    output = tmp_path / "result"

    write_results(rows, output)

    with (output / "graph_coloring_runs.csv").open(newline="") as source:
        raw = list(csv.DictReader(source))
    with (output / "graph_coloring_summary.csv").open(newline="") as source:
        summary = list(csv.DictReader(source))
    assert len(raw) == len(summary) == 2
    assert [int(row["color_count"]) for row in raw] == [2, 2]
    assert [row["seed"] for row in raw] == ["", ""]
    assert [int(row["shuffle_seed"]) for row in raw] == [0, 0]
    for row in summary:
        assert row["runs"] == "1"
        assert all(float(value) == 0 for key, value in row.items() if key.endswith("_std"))


@pytest.mark.parametrize("runs", [0, -1])
def test_nonpositive_run_counts_are_rejected(instance, runs):
    """Require at least one measured run for coloring benchmarks."""
    with pytest.raises(ValueError, match="runs must be at least 1"):
        benchmark_graph_coloring(instance, runs=runs)


@pytest.mark.parametrize(
    "filename, contents, message",
    [
        ("unknown.clq", "p edge 4 0\n", "Expected an FRB filename"),
        ("frb2-2-1.mis", "p edge 4 0\n", "Expected an FRB filename"),
        ("frb0-2-1.clq", "p edge 4 0\n", "Expected an FRB filename"),
        ("frb3-2-1.clq", "p edge 4 0\n", "Vertex count does not match"),
    ],
)
def test_invalid_metadata_is_rejected(tmp_path, filename, contents, message):
    """Reject invalid coloring dataset names and mismatched FRB vertex counts."""
    path = tmp_path / filename
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        benchmark_graph_coloring(path)


@pytest.mark.parametrize(
    "colors",
    [
        {},
        {1: 0, 2: 0, 3: 1},
        {1: 0, 2: 0, 3: 1, 4: 2, 99: 0},
        {1: 0, 2: 0, 3: 0, 4: 0},
        {1: -1, 2: 0, 3: 1, 4: 2},
        {1: 0.5, 2: 0, 3: 1, 4: 2},
    ],
)
def test_invalid_colorings_are_rejected(instance, monkeypatch, colors):
    """Reject incomplete colorings, unknown vertices, edge conflicts, and bad colors."""
    monkeypatch.setattr("src.benchmark.greedy_coloring", lambda graph: colors)

    with pytest.raises(ValueError, match="invalid coloring"):
        benchmark_graph_coloring(instance, runs=1)


def test_counts_distinct_colors_instead_of_largest_label(instance, monkeypatch):
    """Ensure nonconsecutive color labels contribute only their distinct count."""
    monkeypatch.setattr(
        "src.benchmark.greedy_coloring", lambda graph: {1: 10, 2: 10, 3: 99, 4: 42}
    )

    assert benchmark_graph_coloring(instance, runs=1)[0]["color_count"] == 3


def test_cli_finds_nested_coloring_instances_and_writes_artifacts(instance, tmp_path):
    """Check recursive coloring dataset discovery and generated CSV and PNG files."""
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "frb2-2-2.clq").write_bytes(instance.read_bytes())
    output = tmp_path / "result"

    completed = subprocess.run(
        [
            sys.executable,
            "-m", "src.run_benchmark",
            "--problem", "graph-coloring",
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
    with (output / "graph_coloring_runs.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    with (output / "graph_coloring_summary.csv").open(newline="") as source:
        summary = list(csv.DictReader(source))
    assert len(rows) == 8
    assert len(summary) == 4
    assert {row["instance"] for row in rows} == {"frb2-2-1", "frb2-2-2"}
    assert all(row["optimum"] == "2" for row in rows)
    assert {row["seed"] for row in rows} == {""}
    assert {row["shuffle_seed"] for row in rows} == {"7"}
    for suffix in ("runtime", "color_count", "excess", "factor"):
        filename = f"graph_coloring_{suffix}.png"
        assert (output / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_cli_reports_missing_coloring_instances(tmp_path):
    """Check the CLI reports an empty coloring dataset directory clearly."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m", "src.run_benchmark",
            "--problem", "graph-coloring",
            "--data-dir", str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "No .clq instances found" in completed.stderr
