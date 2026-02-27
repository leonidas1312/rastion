# 🧪 Rastion - Your Personal Optimization Lab

A local-first optimization library for portable models and pluggable solvers.
Think Hugging Face for optimization workflows: package problems, swap solvers, benchmark results, and keep run history.

## Why Rastion?

- 🔌 **Plug & Play**: Switch solvers without rewriting problems.
- 📦 **Portable Problems**: Keep model, instances, metadata, and cards together.
- 🚀 **Quantum-Ready**: QUBO and QAOA support included.
- 📊 **Benchmark + History**: Compare solvers and keep reproducible run records.
- 🖥️ **TUI First**: Interactive lab experience with keyboard navigation.

## Installation

Core install:

```bash
pip install -e .
```

TUI install:

```bash
pip install -e ".[tui]"
```

All optional solver extras:

```bash
pip install -e ".[all]"
```

## Launch the Lab

```bash
rastion
```

Main actions:

- 📂 Browse Problems
- ⚙️ Browse Solvers
- ▶️ Solve a Problem
- 📊 Compare Solvers
- 📜 View Run History
- 📦 Export Problem
- 📥 Install Solver (from GitHub URL)

## Built-in Problems

| Problem | Type | Description |
| --- | --- | --- |
| knapsack | MILP | Classic 0-1 knapsack |
| maxcut | QUBO | MaxCut on graph |
| portfolio | QP | Mean-variance portfolio |
| facility_location | MILP | Facility location |
| set_cover | MILP | Set covering |
| tsp | MILP | Traveling salesman |

## Built-in Solvers

| Solver | Type | Best For |
| --- | --- | --- |
| baseline | exact/heuristic | Binary MILP, testing |
| highs | MILP/QP | Fast MILP and QP |
| ortools | MILP | Google's MILP stack |
| scip | MILP | Academic/advanced MILP |
| neal | QUBO | Simulated annealing |
| qaoa | QUBO | Quantum-style QAOA sampling |

## Sharing Solvers

Anyone can create and share a solver plugin.

1. Create a GitHub repo containing `solver.py` at repo root.
2. Share your ZIP URL.
3. Install with:

```bash
rastion install-solver https://github.com/YOU/repo/archive/main.zip
```

Template files:

- `rastion/solvers/solver_template.py`
- `rastion/solvers/solver_card_template.md`

## CLI Commands

- `rastion` (launch TUI)
- `rastion info [--verbose]`
- `rastion validate <spec> <instance>`
- `rastion solve <spec_or_problem> <instance>`
- `rastion benchmark <problem>`
- `rastion install <problem_folder>`
- `rastion export <problem> <destination>`
- `rastion install-solver <zip_url>`

## Documentation

- [Creating Problems](docs/CREATING_PROBLEMS.md)
- [Creating Solvers](docs/CREATING_SOLVERS.md)

## License

Apache-2.0
