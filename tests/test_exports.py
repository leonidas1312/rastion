from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rastion.benchmark import run_benchmark, save_benchmark_results
from rastion.exporters import export_benchmarks_json, export_solvers_json
from rastion.ir.convert import load_problem_json
from rastion.plugins.discovery import discover_solvers
from rastion.solvers.registry import clear_registry


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_problem():
    return load_problem_json(REPO_ROOT / "examples" / "qubo_small.json")


def test_export_benchmarks_writes_normalized_json(tmp_path: Path) -> None:
    clear_registry()
    discover_solvers(plugin_root=REPO_ROOT / "plugins_local", use_entry_points=False, reset_registry=True)
    problem = _load_problem()

    cache_path = tmp_path / "benchmarks-cache.json"
    rows = run_benchmark(problem, repeat=1, iters=40, seed=0)
    save_benchmark_results(rows, problem=problem, cache_path=cache_path)

    out = tmp_path / "benchmarks.json"
    payload = export_benchmarks_json(out, cache_path=cache_path)

    assert out.exists()
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["generated_at"] == payload["generated_at"]
    assert isinstance(parsed["rows"], list)
    assert len(parsed["rows"]) >= 2
    datetime.fromisoformat(str(parsed["generated_at"]))


def test_export_solvers_writes_json(tmp_path: Path) -> None:
    clear_registry()
    discover_solvers(plugin_root=REPO_ROOT / "plugins_local", use_entry_points=False, reset_registry=True)

    out = tmp_path / "solvers.json"
    payload = export_solvers_json(out)

    assert out.exists()
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["count"] == payload["count"]
    assert isinstance(parsed["solvers"], list)
    names = {row["name"] for row in parsed["solvers"]}
    assert {"tabu", "simulated_annealing"}.issubset(names)
