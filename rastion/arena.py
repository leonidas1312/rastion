from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rastion.solvers.registry import RegisteredSolver, get_solver, list_solvers


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _select_solvers(problem: Any, solver_names: list[str] | None = None) -> list[RegisteredSolver]:
    compatible = [
        row
        for row in list_solvers()
        if problem.problem_type in row.solver.supports and problem.n_vars <= row.solver.max_size
    ]

    if solver_names is None:
        return compatible

    wanted = {name.strip() for name in solver_names if name.strip()}
    selected = [row for row in compatible if row.name in wanted]
    return selected


def run_arena(
    problem: Any,
    *,
    solver_names: list[str] | None = None,
    iters: int = 2_000,
    time_budget_ms: int | None = None,
    seed: int = 0,
    emit_every: int = 50,
) -> dict[str, object]:
    selected = _select_solvers(problem, solver_names)

    out_solvers: list[dict[str, object]] = []
    for idx, row in enumerate(selected):
        solver = get_solver(row.name)
        solver_seed = seed + idx

        run_kwargs = {
            "iters": iters,
            "seed": solver_seed,
            "emit_every": emit_every,
        }
        if time_budget_ms is not None:
            run_kwargs["time_budget_ms"] = int(time_budget_ms)

        try:
            events: list[dict[str, float | int]] = []
            best_value = float("inf")

            started_stream = time.perf_counter()
            for event in solver.solve_stream(problem, **run_kwargs):
                event_row = {
                    "t_ms": int(event.t_ms),
                    "iter": int(event.iter),
                    "best_value": float(event.best_value),
                }
                events.append(event_row)
                best_value = min(best_value, float(event.best_value))
            stream_runtime_ms = max(int((time.perf_counter() - started_stream) * 1000.0), 0)

            solution = None
            route: list[int] | None = None
            if str(getattr(problem, "problem_type", "")) == "tsp":
                started_solve = time.perf_counter()
                solution = solver.solve(problem, **run_kwargs)
                solve_runtime_ms = max(int((time.perf_counter() - started_solve) * 1000.0), 0)
                stream_runtime_ms = solve_runtime_ms
                best_value = float(solution.best_value)
                route_meta = solution.metadata.get("route")
                if isinstance(route_meta, list):
                    route = [int(node) for node in route_meta]
                else:
                    route = [int(node) for node in solution.best_x.tolist()]
            elif best_value == float("inf"):
                solution = solver.solve(problem, **run_kwargs)
                best_value = float(solution.best_value)
                events = [{"t_ms": stream_runtime_ms, "iter": int(iters), "best_value": best_value}]

            if not events:
                events = [{"t_ms": int(stream_runtime_ms), "iter": int(iters), "best_value": float(best_value)}]

            final_payload: dict[str, float | int | list[int]] = {
                "best_value": float(best_value),
                "runtime_ms": int(stream_runtime_ms),
            }
            if route is not None:
                final_payload["route"] = route

            out_solvers.append(
                {
                    "name": row.name,
                    "metadata": row.metadata.model_dump(),
                    "events": events,
                    "final": final_payload,
                }
            )
        except Exception as exc:  # noqa: BLE001
            out_solvers.append(
                {
                    "name": row.name,
                    "metadata": row.metadata.model_dump(),
                    "events": [],
                    "final": {
                        "best_value": None,
                        "runtime_ms": 0,
                    },
                    "error": str(exc),
                }
            )

    problem_payload: dict[str, object] = {
        "name": problem.name,
        "type": problem.problem_type,
        "n_vars": int(problem.n_vars),
    }
    if hasattr(problem, "depot"):
        problem_payload["depot"] = int(getattr(problem, "depot"))
    if hasattr(problem, "coords"):
        coords = getattr(problem, "coords")
        nodes = [{"id": idx, "x": float(coords[idx, 0]), "y": float(coords[idx, 1])} for idx in range(int(problem.n_vars))]
        problem_payload["nodes"] = nodes

    return {
        "generated_at": _now_utc_iso(),
        "problem": problem_payload,
        "solvers": out_solvers,
    }


def write_arena_json(
    problem: Any,
    out_path: str | Path,
    *,
    solver_names: list[str] | None = None,
    iters: int = 2_000,
    time_budget_ms: int | None = None,
    seed: int = 0,
    emit_every: int = 50,
) -> dict[str, object]:
    payload = run_arena(
        problem,
        solver_names=solver_names,
        iters=iters,
        time_budget_ms=time_budget_ms,
        seed=seed,
        emit_every=emit_every,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
