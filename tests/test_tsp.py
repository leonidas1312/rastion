from __future__ import annotations

from pathlib import Path

from rastion.plugins.discovery import discover_solvers
from rastion.solvers.registry import clear_registry, list_solvers
from rastion.tsp.arena import build_tsp_arena_bundle
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
        rows = instance["solvers"]
        assert isinstance(rows, list)
        assert len(rows) >= 2
        for row in rows:
            assert "final" in row
            if "error" not in row:
                route = row["final"].get("route")
                assert isinstance(route, list)
                assert len(route) >= int(instance["n_vars"])
