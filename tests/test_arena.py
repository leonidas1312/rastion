from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rastion.arena import run_arena, write_arena_json
from rastion.ir.convert import load_problem_json
from rastion.plugins.discovery import discover_solvers
from rastion.solvers.registry import clear_registry


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
        assert len(events) > 1
        assert all("best_value" in event for event in events)


def test_write_arena_json_writes_output_file(tmp_path: Path) -> None:
    clear_registry()
    discover_solvers(plugin_root=REPO_ROOT / "plugins_local", use_entry_points=False, reset_registry=True)
    problem = _load_problem()

    out = tmp_path / "arena.json"
    payload = write_arena_json(problem, out, iters=80, seed=1, emit_every=10)

    assert out.exists()
    assert payload["problem"]["name"] == problem.name
