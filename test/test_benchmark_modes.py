"""Check CSV-only benchmarking and plotting from saved raw CSVs."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from src.benchmark import (
    benchmark_graph_coloring,
    benchmark_vertex_cover,
    read_results,
    write_results,
)


VALID_CSV = (
    "instance,run,shuffle_seed,vertices,edges,optimum,algorithm,cover_size,runtime_seconds,excess,factor\n"
    "frb2-2-1,1,0,4,3,2,approx,3,0.25,1,1.5\n"
    "frb2-2-1,1,0,4,3,2,randomized,2,0.5,0,1\n"
)


@pytest.fixture
def data_dir(tmp_path):
    """Create a directory with matching cover and coloring DIMACS datasets."""
    directory = tmp_path / "data"
    directory.mkdir()
    for suffix in ("mis", "clq"):
        (directory / f"frb2-2-1.{suffix}").write_text(
            "p edge 4 3\ne 1 2\ne 2 3\ne 3 4\n", encoding="utf-8"
        )
    return directory


@pytest.fixture
def saved_results(data_dir, tmp_path):
    """Write repeatable raw and summary CSVs for both benchmark problems."""
    output = tmp_path / "result"
    for suffix, benchmark in (
        ("mis", benchmark_vertex_cover), ("clq", benchmark_graph_coloring)
    ):
        write_results(benchmark(data_dir / f"frb2-2-1.{suffix}", runs=2, seed=7), output)
    return output


def run_cli(*arguments):
    """Run the public benchmark module and return captured output and exit status."""
    return subprocess.run(
        [sys.executable, "-m", "run_benchmark", *map(str, arguments)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("problem", ["all", "vertex-cover", "graph-coloring"])
def test_csv_mode_writes_only_csvs_and_preserves_existing_plots(
    data_dir, tmp_path, problem
):
    """Verify CSV mode writes selected measurements while preserving existing PNGs."""
    output = tmp_path / "result"
    output.mkdir()
    selected = (
        ("vertex_cover", "graph_coloring") if problem == "all"
        else (problem.replace("-", "_"),)
    )
    existing_plot = output / f"{selected[0]}_runtime.png"
    existing_plot.write_bytes(b"existing plot")

    completed = run_cli(
        "--mode", "csv", "--problem", problem,
        "--data-dir", data_dir, "--output-dir", output,
        "--runs", "2", "--seed", "7",
        "--warmup", "3",
    )

    assert completed.returncode == 0, completed.stderr
    assert "Saved CSVs to" in completed.stdout
    assert "3 warm-up runs and 2 measured runs per algorithm" in completed.stdout
    assert existing_plot.read_bytes() == b"existing plot"
    assert {path.name for path in output.iterdir()} == {existing_plot.name} | {
        f"{name}_{kind}.csv" for name in selected for kind in ("runs", "summary")
    }
    for name in selected:
        with (output / f"{name}_runs.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 4
        expected_seeds = ["", "7", "", "8"] if name == "vertex_cover" else [""] * 4
        assert [row["seed"] for row in rows] == expected_seeds
        assert {row["shuffle_seed"] for row in rows} == {"7"}


@pytest.mark.parametrize("problem", ["all", "vertex-cover", "graph-coloring"])
def test_plot_mode_uses_saved_csvs_without_datasets_or_csv_writes(
    saved_results, tmp_path, problem
):
    """Check plot mode needs only selected raw CSVs and leaves their bytes intact."""
    selected = (
        ("vertex_cover", "graph_coloring") if problem == "all"
        else (problem.replace("-", "_"),)
    )
    for path in saved_results.glob("*.csv"):
        if not any(path.name.startswith(name) for name in selected):
            path.unlink()
    csv_bytes = {path.name: path.read_bytes() for path in saved_results.glob("*.csv")}

    completed = run_cli(
        "--mode", "plot", "--problem", problem,
        "--output-dir", saved_results,
        "--data-dir", tmp_path / "unavailable-data", "--runs", "0",
        "--warmup", "-1",
    )

    assert completed.returncode == 0, completed.stderr
    assert "Saved plots from CSVs" in completed.stdout
    assert "Benchmarking" not in completed.stdout
    assert {
        path.name: path.read_bytes() for path in saved_results.glob("*.csv")
    } == csv_bytes
    expected_plots = set()
    for name in selected:
        suffixes = ["runtime", "color_count" if name == "graph_coloring" else "size", "excess", "factor"]
        if name == "vertex_cover":
            suffixes.append("std")
        expected_plots.update(f"{name}_{suffix}.png" for suffix in suffixes)
    assert {path.name for path in saved_results.glob("*.png")} == expected_plots
    for filename in expected_plots:
        assert (saved_results / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_plot_mode_checks_all_csvs_before_drawing(saved_results):
    """Ensure a missing requested CSV prevents any plot from being replaced."""
    (saved_results / "graph_coloring_runs.csv").unlink()
    existing_plot = saved_results / "vertex_cover_runtime.png"
    existing_plot.write_bytes(b"existing plot")

    completed = run_cli("--mode", "plot", "--output-dir", saved_results)

    assert completed.returncode == 2
    assert "graph_coloring_runs.csv" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert existing_plot.read_bytes() == b"existing plot"
    assert list(saved_results.glob("*.png")) == [existing_plot]


def test_plot_mode_rejects_wrong_problem_csv(saved_results):
    """Reject cover rows supplied as coloring results before creating plots."""
    (saved_results / "graph_coloring_runs.csv").write_bytes(
        (saved_results / "vertex_cover_runs.csv").read_bytes()
    )

    completed = run_cli("--mode", "plot", "--output-dir", saved_results)

    assert completed.returncode == 2
    assert "Expected color_count results" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not list(saved_results.glob("*.png"))


@pytest.mark.parametrize("coloring", [False, True])
def test_read_results_converts_numeric_plot_fields(tmp_path, coloring):
    """Verify raw cover and coloring CSV values become numeric plotting fields."""
    path = tmp_path / "runs.csv"
    contents = VALID_CSV
    if coloring:
        contents = contents.replace("cover_size", "color_count")
        contents = contents.replace("approx", "greedy").replace(
            "randomized", "smallest_last"
        )
    path.write_text(contents, encoding="utf-8")

    rows = read_results(path)

    size = "color_count" if coloring else "cover_size"
    assert len(rows) == 2
    assert [row[size] for row in rows] == [3, 2]
    assert all(
        isinstance(row[key], int)
        for row in rows for key in ("vertices", "edges", "optimum", size)
    )
    assert [row["runtime_seconds"] for row in rows] == [0.25, 0.5]
    assert [row["excess"] for row in rows] == [1, 0]
    assert [row["factor"] for row in rows] == [1.5, 1]


@pytest.mark.parametrize(
    "contents, message",
    [
        ("", "Expected cover_size or color_count"),
        (VALID_CSV.splitlines()[0] + "\n", "No benchmark results"),
        (VALID_CSV.replace("cover_size", "size"), "Expected cover_size or color_count"),
        (VALID_CSV.replace("shuffle_seed", "old_seed"), "regenerate CSVs with --mode csv"),
        (VALID_CSV.replace("runtime_seconds", "duration"), "Missing CSV columns"),
        (VALID_CSV.replace(",0.25,", ",invalid,"), "Invalid numeric value"),
        (VALID_CSV.replace(",0.25,", ",nan,"), "Invalid benchmark result"),
        (VALID_CSV.replace("approx", "unknown"), "Invalid benchmark result"),
        ("\n".join(VALID_CSV.splitlines()[:2]) + "\n", "Expected both algorithms"),
    ],
)
def test_read_results_reports_invalid_csvs(tmp_path, contents, message):
    """Reject missing columns, invalid values, and incomplete algorithm groups."""
    path = tmp_path / "runs.csv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_results(path)


def test_read_results_rejects_reshuffling_between_runs(tmp_path):
    """Require one fixed input shuffle seed for all runs of an instance."""
    path = tmp_path / "runs.csv"
    path.write_text(
        VALID_CSV + VALID_CSV.splitlines()[1].replace(",1,0,", ",2,1,") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="one fixed shuffle_seed"):
        read_results(path)


def test_read_results_rejects_varying_deterministic_results(tmp_path):
    """Reject changing deterministic solution sizes for a fixed instance."""
    path = tmp_path / "runs.csv"
    path.write_text(
        VALID_CSV + "frb2-2-1,2,0,4,3,2,approx,4,0.3,2,2\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Deterministic solution sizes vary"):
        read_results(path)
