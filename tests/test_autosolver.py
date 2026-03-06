from pathlib import Path

import numpy as np

from rastion.ir.convert import load_problem_json
from rastion.plugins.discovery import discover_solvers
from rastion.plugins.registry import clear_registry, register_solver
from rastion.plugins.schema import SolverMetadata
from rastion.solvers.autosolver import AutoSolver
from rastion.solvers.base import Solution, Solver


REPO_ROOT = Path(__file__).resolve().parents[1]


class FailingSolver(Solver):
    name = "failing"
    supports = ["qubo"]
    max_size = 10_000
    hardware = ["cpu"]
    quality = 1.0

    def solve(self, problem_ir, **kwargs):  # type: ignore[override]
        raise RuntimeError("intentional failure")


def _load_problem():
    return load_problem_json(REPO_ROOT / "examples" / "qubo_small.json")


def test_autosolver_selects_compatible_solver_and_returns_solution(tmp_path: Path) -> None:
    clear_registry()
    discover_solvers(plugin_root=REPO_ROOT / "plugins_local", use_entry_points=False)
    problem = _load_problem()

    auto = AutoSolver(benchmark_cache_path=tmp_path / "benchmarks.json")
    solution = auto.solve(problem, iters=80, seed=7)

    assert solution.solver_name in {"tabu", "simulated_annealing"}
    assert solution.best_x.shape == (problem.n_vars,)
    assert isinstance(solution.best_value, float)
    assert solution.selection_reason is not None


def test_autosolver_falls_back_when_first_solver_fails(tmp_path: Path) -> None:
    clear_registry()
    discover_solvers(plugin_root=REPO_ROOT / "plugins_local", use_entry_points=False)

    register_solver(
        FailingSolver(),
        SolverMetadata(
            name="failing",
            supports=["qubo"],
            max_size=10_000,
            hardware=["cpu"],
            quality=1.0,
            source="test",
        ),
    )

    problem = _load_problem()
    auto = AutoSolver(w1=0.1, w2=0.8, w3=0.1, benchmark_cache_path=tmp_path / "benchmarks.json")
    solution = auto.solve(problem, iters=40, seed=1)

    assert solution.solver_name in {"tabu", "simulated_annealing"}
    assert "fallbacks=" in (solution.selection_reason or "")
