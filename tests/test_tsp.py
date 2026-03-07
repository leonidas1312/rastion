from __future__ import annotations

from pathlib import Path

import pytest

from rastion.plugins.discovery import discover_solvers
from rastion.solvers.registry import clear_registry, get_solver, list_solvers
from rastion.tsp.arena import build_tsp_arena_bundle
from rastion.tsp.references import gap_to_reference, get_tsplib_reference
from rastion.tsp.tsplib import default_tsplib_paths, load_tsplib_problem


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_tsplib_problem_parses_berlin52() -> None:
    problem = load_tsplib_problem(REPO_ROOT / "examples" / "tsplib" / "berlin52.tsp")
    assert problem.problem_type == "tsp"
    assert problem.n_vars == 52
    assert problem.depot == 0
    assert problem.coords.shape == (52, 2)


def test_tsp_solvers_are_discoverable_without_breaking_qubo_plugins() -> None:
    clear_registry()
    discover_solvers(plugin_root=REPO_ROOT / "plugins_local", use_entry_points=False, reset_registry=True)
    names = {row.name for row in list_solvers()}

    assert "simulated_annealing" in names
    assert "tabu" in names
    assert "tsp_nearest_neighbor" in names
    assert "tsp_two_opt" in names


def test_tsplib_reference_lookup_and_gap_calculation() -> None:
    berlin = get_tsplib_reference("berlin52")
    ch = get_tsplib_reference("ch150")
    a280 = get_tsplib_reference("a280")

    assert berlin is not None
    assert berlin.best_known_distance == 7542.0
    assert ch is not None
    assert ch.best_known_distance == 6528.0
    assert a280 is not None
    assert a280.best_known_distance == 2579.0
    assert gap_to_reference(8154.0, berlin.best_known_distance) == pytest.approx(8.1145584725537)
    assert gap_to_reference(None, berlin.best_known_distance) is None


def test_build_tsp_arena_bundle_has_three_sizes_and_routes() -> None:
    clear_registry()
    discover_solvers(plugin_root=REPO_ROOT / "plugins_local", use_entry_points=False, reset_registry=True)

    payload = build_tsp_arena_bundle(
        tsplib_dir=REPO_ROOT / "examples" / "tsplib",
        solver_names=["tsp_nearest_neighbor", "tsp_two_opt"],
        iters=300,
        seed=0,
        emit_every=25,
    )

    instances = payload.get("instances", [])
    assert isinstance(instances, list)
    assert len(instances) == len(default_tsplib_paths(REPO_ROOT / "examples" / "tsplib"))

    for instance in instances:
        assert instance["size"] in {"small", "medium", "large"}
        assert int(instance["n_vars"]) > 10
        assert instance["reference"] is not None
        assert instance["reference"]["label"] == "Best known"
        rows = instance["solvers"]
        assert isinstance(rows, list)
        assert len(rows) >= 2
        for row in rows:
            assert "final" in row
            assert "gap_to_reference_pct" in row
            if "error" not in row:
                route = row["final"].get("route")
                assert isinstance(route, list)
                assert len(route) >= int(instance["n_vars"])


def test_nearest_neighbor_trace_uses_final_tour_metric() -> None:
    clear_registry()
    discover_solvers(plugin_root=REPO_ROOT / "plugins_local", use_entry_points=False, reset_registry=True)
    solver = get_solver("tsp_nearest_neighbor")
    problem = load_tsplib_problem(REPO_ROOT / "examples" / "tsplib" / "berlin52.tsp")

    solution, events = solver.solve_trace(problem, seed=0)

    assert len(events) == 1
    assert events[0].best_value == solution.best_value
    assert events[0].iter == problem.n_vars


def test_ortools_solver_runs_without_random_seed_error() -> None:
    pytest.importorskip("ortools")
    clear_registry()
    discover_solvers(plugin_root=REPO_ROOT / "plugins_local", use_entry_points=False, reset_registry=True)
    solver = get_solver("tsp_ortools")
    problem = load_tsplib_problem(REPO_ROOT / "examples" / "tsplib" / "berlin52.tsp")

    solution, events = solver.solve_trace(problem, seed=0, time_budget_ms=250)

    assert solution.best_value > 0
    assert len(events) == 1
    assert "route" in solution.metadata
