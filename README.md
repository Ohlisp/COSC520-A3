# COSC 520 Assignment 3: vertex cover and graph coloring

Python implementations of a deterministic 2-approximation and a randomized
algorithm for vertex cover, plus greedy and smallest-last graph coloring.
.

## Project layout

```text
.
├── README.md
├── pyproject.toml
├── flake.nix
├── flake.lock
├── run_benchmark.py         # Benchmark entry point
├── src/
│   ├── __init__.py
│   ├── graph.py             # Undirected graph and DIMACS reader
│   ├── vertex_cover.py      # Deterministic and randomized vertex cover
│   ├── graph_coloring.py    # Greedy and smallest-last coloring
│   ├── benchmark.py         # Measurements, summaries, and CSV handling
│   ├── run_benchmark.py     # Command-line options and plots
│   └── demo.py              # Examples of all four algorithms
├── test/                    # Unit tests
├── data/
│   ├── frb-vertex-cover/    # Vertex-cover instances (.mis)
│   └── frb-gragh-coloring/  # Graph-coloring instances (.clq)
└── result/                  # Benchmark outputs
```

## Setup

Run these commands from the project root. Choose either setup below.

### Nix

With Nix and flakes enabled:

```sh
nix develop path:.
```



### Python virtual environment

Use Python 3.10 or later. In a macOS, Linux, or Windows WSL2 terminal:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

Activate the virtual environment again when opening a new shell. The editable
install picks up source changes immediately.

## Dataset

No extra dataset download is required. The repository includes the benchmark datasets under `data/`.


## Tests and demo

After either setup:

```sh
python -m pytest -q
python -m src.demo
```

The demo prints an 8-vertex cube graph with 12 edges and degree 3 at every vertex,
and checks every result. Text diagrams mark selected cover vertices with `*`
and show color assignments as `vertex:color`.

## Benchmark

Run the full benchmark:

```sh
python -m run_benchmark
```

the benchmark CSVs and plots will be written to `result/`. For more details, run
`python -m run_benchmark --help`. Inside the Nix shell, `run-benchmark` accepts
the same options as `python -m run_benchmark`.
