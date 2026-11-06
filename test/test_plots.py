"""Check separate quality plots and within-instance randomized variability."""

from math import sqrt

import pytest
from matplotlib.container import BarContainer

from src import run_benchmark


@pytest.fixture
def figures(monkeypatch):
    """Capture figures created by the plotting module for inspection after saving."""
    captured = []
    subplots = run_benchmark.plt.subplots

    def capture(*args, **kwargs):
        """Record each new figure and return the original subplot result."""
        figure, axes = subplots(*args, **kwargs)
        captured.append(figure)
        return figure, axes

    monkeypatch.setattr(run_benchmark.plt, "subplots", capture)
    return captured


@pytest.mark.parametrize("coloring", [False, True])
def test_runtime_size_excess_and_ratio_plots(tmp_path, figures, coloring):
    """Check family means, OPT baselines, plot labels, and obsolete plot removal."""
    algorithms = ("greedy", "smallest_last") if coloring else ("approx", "randomized")
    problem = "graph_coloring" if coloring else "vertex_cover"
    legacy = tmp_path / f"{problem}.png"
    legacy.write_bytes(b"old combined figure")
    size_metric = "color_count" if coloring else "cover_size"
    rows = []
    for index, algorithm in enumerate(algorithms):
        counts = (2, 4, 4, 3) if index == 0 else (2, 3, 3, 2)
        for sample, ((vertices, edges, runtime), count) in enumerate(zip(
            ((4, 3, 0.1), (4, 3, 0.3), (4, 4, 0.6), (5, 2, 0.8)), counts
        )):
            rows.append(
                {
                    "algorithm": algorithm,
                    "instance": f"instance-{sample}",
                    "vertices": vertices,
                    "edges": edges,
                    "optimum": 2,
                    size_metric: count,
                    "runtime_seconds": runtime + index * 0.2,
                    "excess": count - 2,
                    "factor": count / 2,
                }
            )

    run_benchmark.plot_results(rows, tmp_path)

    assert not legacy.exists()
    assert len(figures) == (4 if coloring else 5)
    assert all(len(figure.axes) == 1 for figure in figures)
    runtime = figures[0].axes[0]
    assert runtime.get_xlabel() == "Average graph size |V| + |E|"
    assert runtime.get_ylabel() == "Average runtime (seconds)"
    assert len(runtime.lines) == 2
    for index, line in enumerate(runtime.lines):
        assert list(line.get_xdata()) == pytest.approx([7, 22 / 3])
        assert list(line.get_ydata()) == pytest.approx(
            [0.8 + index * 0.2, 1 / 3 + index * 0.2]
        )

    comparison = figures[1].axes[0]
    bars = [
        container for container in comparison.containers
        if isinstance(container, BarContainer)
    ]
    assert len(bars) == 3
    assert bars[0].get_label() == "OPT"
    assert [bar.get_height() for bar in bars[0]] == [2, 2]
    assert [bar.get_height() for bar in bars[1]] == pytest.approx([10 / 3, 3])
    assert [bar.get_height() for bar in bars[2]] == pytest.approx([8 / 3, 2])
    assert comparison.get_xlabel() == "Vertices |V|"
    assert [label.get_text() for label in comparison.get_xticklabels()] == ["4", "5"]
    assert "3.3" in {text.get_text() for text in comparison.texts}
    assert all(container.errorbar is None for container in bars)
    assert not any("std" in text.get_text().lower() for text in comparison.texts)

    for figure, expected, baseline in (
        (figures[2], ([4 / 3, 1], [2 / 3, 0]), 0),
        (figures[3], ([5 / 3, 1.5], [4 / 3, 1]), 1),
    ):
        axis = figure.axes[0]
        assert axis.get_xlabel() == "Vertices |V|"
        for line, values in zip(axis.lines[:2], expected):
            assert list(line.get_xdata()) == [4, 5]
            assert list(line.get_ydata()) == pytest.approx(values)
        assert list(axis.lines[2].get_ydata()) == [baseline, baseline]

    if coloring:
        assert not list(tmp_path.glob("graph_coloring*std*"))
    else:
        assert len(figures[4].axes[0].lines) == 1
        assert list(figures[4].axes[0].lines[0].get_ydata()) == [0, 0]


def test_runtime_has_one_point_per_family_across_five_instances(tmp_path, figures):
    """Verify runtime points average all instances and repeated runs in a family."""
    rows = []
    for index, algorithm in enumerate(("approx", "randomized")):
        for vertices, edges, runtime in ((10, 20, 0.1), (20, 60, 0.2)):
            optimum = vertices - vertices // 5
            for instance in range(5):
                for run in range(2):
                    rows.append(
                        {
                            "instance": f"frb{vertices // 5}-5-{instance + 1}",
                            "algorithm": algorithm,
                            "vertices": vertices,
                            "edges": edges + instance * 2,
                            "optimum": optimum,
                            "cover_size": optimum + 1,
                            "runtime_seconds": (
                                runtime + instance * 0.01
                                + run * 0.002 + index * 0.05
                            ),
                            "excess": 1,
                            "factor": (optimum + 1) / optimum,
                        }
                    )

    run_benchmark.plot_results(rows, tmp_path)

    for index, line in enumerate(figures[0].axes[0].lines):
        assert list(line.get_xdata()) == [34, 84]
        assert list(line.get_ydata()) == pytest.approx(
            [0.121 + index * 0.05, 0.221 + index * 0.05]
        )


def test_randomized_std_averages_instance_stds_instead_of_pooling(tmp_path, figures):
    """Check variation is averaged from per-instance sample standard deviations."""
    rows = []
    for vertices, instance, counts in (
        (4, "instance-a", (2, 4)),
        (4, "instance-b", (4, 4)),
        (5, "instance-c", (3,)),
    ):
        for algorithm in ("approx", "randomized"):
            for count in counts:
                size = count if algorithm == "randomized" else counts[0]
                rows.append(
                    {
                        "algorithm": algorithm, "instance": instance,
                        "vertices": vertices, "edges": 3, "optimum": 2,
                        "cover_size": size, "runtime_seconds": 0.1,
                        "excess": size - 2, "factor": size / 2,
                    }
                )

    run_benchmark.plot_results(rows, tmp_path)

    axis = figures[-1].axes[0]
    assert axis.get_ylabel() == "Mean per-instance STD of cover size"
    assert len(axis.lines) == 1
    assert axis.lines[0].get_label() == "Randomized"
    assert list(axis.lines[0].get_xdata()) == [4, 5]
    assert list(axis.lines[0].get_ydata()) == pytest.approx([sqrt(2) / 2, 0])
