from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from rastion.ir.types import ProblemIR
from rastion.solvers.registry import list_solvers


DEFAULT_BENCHMARK_PATH = Path(".rastion/benchmarks.json")


def _benchmark_key(problem_type: str, n_vars: int, solver_name: str) -> str:
    return f"{problem_type}|{n_vars}|{solver_name}"


def run_benchmark(problem: ProblemIR, repeat: int = 3, **solver_kwargs: int | float) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for registered in list_solvers():
        solver = registered.solver
        if problem.problem_type not in solver.supports or problem.n_vars > solver.max_size:
            continue

        best_value = float("inf")
        total_runtime = 0.0
        failed = False
        for _ in range(repeat):
            started = time.perf_counter()
            try:
                solution = solver.solve(problem, **solver_kwargs)
            except Exception:  # noqa: BLE001
                failed = True
                break
            total_runtime += time.perf_counter() - started
            if solution.best_value < best_value:
                best_value = solution.best_value

        if failed:
            continue

        rows.append(
            {
                "solver_name": solver.name,
                "best_value": best_value,
                "avg_runtime": total_runtime / repeat,
            }
        )

    rows.sort(key=lambda r: (float(r["best_value"]), float(r["avg_runtime"])))
    return rows


def save_benchmark_results(
    rows: list[dict[str, float | str]],
    *,
    problem: ProblemIR,
    cache_path: str | Path = DEFAULT_BENCHMARK_PATH,
) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        payload = {"entries": {}}

    if not rows:
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    by_value = sorted(rows, key=lambda r: float(r["best_value"]))
    by_runtime = sorted(rows, key=lambda r: float(r["avg_runtime"]))
    n_rows = len(rows)
    denom = max(n_rows - 1, 1)

    value_rank = {str(row["solver_name"]): i for i, row in enumerate(by_value)}
    speed_rank = {str(row["solver_name"]): i for i, row in enumerate(by_runtime)}

    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        solver_name = str(row["solver_name"])
        value = float(row["best_value"])
        runtime = float(row["avg_runtime"])
        value_score = 1.0 - (value_rank[solver_name] / denom)
        speed_score = 1.0 - (speed_rank[solver_name] / denom)
        hint = 0.7 * value_score + 0.3 * speed_score

        payload["entries"][_benchmark_key(problem.problem_type, problem.n_vars, solver_name)] = {
            "hint": float(min(max(hint, 0.0), 1.0)),
            "best_value": value,
            "avg_runtime": runtime,
            "updated_at": now,
        }

    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
