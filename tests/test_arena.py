from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from rastion.arena import run_arena, write_arena_json
from rastion.ir.convert import load_problem_json
from rastion.plugins.discovery import discover_solvers
from rastion.solvers.base import ProgressEvent, Solution, Solver
from rastion.solvers.registry import register_solver
from rastion.solvers.schema import SolverMetadata
from rastion.solvers.registry import clear_registry
from rastion.tsp.tsplib import load_tsplib_problem


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_problem():
    return load_problem_json(REPO_ROOT / "examples" / "qubo_small.json")


def test_arena_payload_contains_streaming_events_for_local_plugins(tmp_path: Path) -> None:
    clear_registry()
    discover_solvers(plugin_root=REPO_ROOT / "plugins_local", use_entry_points=False, reset_registry=True)
    problem = _load_problem()

    payload = run_arena(problem, iters=120, seed=0, emit_every=20)

    assert "generated_at" in payload
    datetime.fromisoformat(str(payload["generated_at"]))

    rows = payload.get("solvers", [])
    assert isinstance(rows, list)
    names = {str(row["name"]) for row in rows}
    assert {"tabu", "simulated_annealing"}.issubset(names)

    by_name = {str(row["name"]): row for row in rows}
    for required in ("tabu", "simulated_annealing"):
        events = by_name[required]["events"]
        assert isinstance(events, list)
        assert len(events) >= 1
        assert all("best_value" in event for event in events)


def test_write_arena_json_writes_output_file(tmp_path: Path) -> None:
    clear_registry()
    discover_solvers(plugin_root=REPO_ROOT / "plugins_local", use_entry_points=False, reset_registry=True)
    problem = _load_problem()

    out = tmp_path / "arena.json"
    payload = write_arena_json(problem, out, iters=80, seed=1, emit_every=10)

    assert out.exists()
    assert payload["problem"]["name"] == problem.name


class _TraceOnlySolver(Solver):
    name = "trace_only"
    supports = ["tsp"]
    max_size = 100
    hardware = ["cpu"]

    def solve(self, problem_ir, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("run_arena should not call solve() when solve_trace() is available")

    def solve_trace(self, problem_ir, **kwargs):
        solution = Solution(
            solver_name=self.name,
            best_x=np.asarray([0, 1, 2, 0], dtype=int),
            best_value=123.0,
            metadata={"route": [0, 1, 2, 0], "runtime_ms": 9},
        )
        return solution, [ProgressEvent(t_ms=9, iter=7, best_value=123.0)]


class _SolveOnlySolver(Solver):
    name = "solve_only"
    supports = ["qubo"]
    max_size = 64
    hardware = ["cpu"]

    def solve(self, problem_ir, **kwargs):
        return Solution(
            solver_name=self.name,
            best_x=np.asarray([1, 0, 1], dtype=int),
            best_value=-5.0,
            metadata={"runtime_ms": 13},
        )


def test_solver_default_solve_trace_wraps_solve() -> None:
    solver = _SolveOnlySolver()
    solution, events = solver.solve_trace(_load_problem(), iters=80)

    assert solution.best_value == -5.0
    assert len(events) == 1
    assert events[0].t_ms == 13
    assert events[0].iter == 80
    assert events[0].best_value == -5.0


def test_run_arena_uses_single_trace_for_final_result() -> None:
    clear_registry()
    register_solver(
        "trace_only",
        _TraceOnlySolver(),
        SolverMetadata(name="trace_only", supports=["tsp"], max_size=100, hardware=["cpu"], quality=0.5, source="test"),
    )
    problem = load_tsplib_problem(REPO_ROOT / "examples" / "tsplib" / "berlin52.tsp")

    payload = run_arena(
        problem=problem,
        solver_names=["trace_only"],
        iters=50,
        seed=0,
        emit_every=10,
    )

    rows = payload["solvers"]
    assert len(rows) == 1
    assert rows[0]["events"] == [{"t_ms": 9, "iter": 7, "best_value": 123.0}]
    assert rows[0]["final"] == {"best_value": 123.0, "runtime_ms": 9, "route": [0, 1, 2, 0]}
