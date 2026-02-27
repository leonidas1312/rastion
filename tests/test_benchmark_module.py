import json
from pathlib import Path

from rastion.benchmark import compare, run_benchmark_suite
from rastion.registry.manager import init_registry, runs_file


def test_compare_runs_baseline_on_knapsack(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "rastion-home"
    monkeypatch.setenv("RASTION_HOME", str(home))
    init_registry()

    results = compare("knapsack", instance="default", solvers=["baseline"], time_limit="5s", runs=2)
    assert len(results) == 1

    result = results[0]
    assert result.solver == "baseline"
    assert result.runtime >= 0.0
    assert result.status in {"OPTIMAL", "FEASIBLE", "UNKNOWN", "ERROR"}


def test_benchmark_suite_can_export_csv_and_json(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "rastion-home"
    monkeypatch.setenv("RASTION_HOME", str(home))
    init_registry()

    suite = run_benchmark_suite(
        problems=["knapsack"],
        solvers=["baseline"],
        time_limit="5s",
        runs=1,
    )

    csv_path = suite.to_csv(tmp_path / "bench.csv")
    json_path = suite.to_json(tmp_path / "bench.json")

    assert csv_path.exists()
    assert json_path.exists()
    assert suite.rows


def test_compare_saves_runs_to_history_by_default(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "rastion-home"
    monkeypatch.setenv("RASTION_HOME", str(home))
    init_registry()

    compare("knapsack", instance="default", solvers=["baseline"], time_limit="5s", runs=2)

    history = runs_file()
    lines = [line for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 2
    assert all(json.loads(line).get("solver_name") == "baseline" for line in lines[-2:])


def test_compare_can_skip_history_persistence(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "rastion-home"
    monkeypatch.setenv("RASTION_HOME", str(home))
    init_registry()

    compare("knapsack", instance="default", solvers=["baseline"], time_limit="5s", runs=1, save_to_history=False)

    history = runs_file()
    assert history.read_text(encoding="utf-8").strip() == ""
