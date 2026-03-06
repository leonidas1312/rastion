from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rastion.arena import run_arena
from rastion.tsp.tsplib import default_tsplib_paths, load_tsplib_problem


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_tsp_arena_bundle(
    *,
    tsplib_dir: str | Path = "examples/tsplib",
    solver_names: list[str] | None = None,
    iters: int = 2_000,
    time_budget_ms: int | None = None,
    seed: int = 0,
    emit_every: int = 50,
) -> dict[str, object]:
    instances: list[dict[str, object]] = []

    for offset, (size_label, path) in enumerate(default_tsplib_paths(tsplib_dir)):
        problem = load_tsplib_problem(path)
        arena_payload = run_arena(
            problem,
            solver_names=solver_names,
            iters=iters,
            time_budget_ms=time_budget_ms,
            seed=seed + offset * 100,
            emit_every=emit_every,
        )

        nodes = [
            {
                "id": idx,
                "x": float(problem.coords[idx, 0]),
                "y": float(problem.coords[idx, 1]),
            }
            for idx in range(problem.n_vars)
        ]

        instances.append(
            {
                "id": problem.name,
                "size": size_label,
                "name": problem.name,
                "type": problem.problem_type,
                "n_vars": problem.n_vars,
                "depot": problem.depot,
                "source": str(path),
                "nodes": nodes,
                "solvers": arena_payload.get("solvers", []),
                "generated_at": arena_payload.get("generated_at"),
            }
        )

    return {
        "generated_at": _now_utc_iso(),
        "suite": "tsplib",
        "instances": instances,
    }


def write_tsp_arena_bundle(
    out_path: str | Path,
    *,
    tsplib_dir: str | Path = "examples/tsplib",
    solver_names: list[str] | None = None,
    iters: int = 2_000,
    time_budget_ms: int | None = None,
    seed: int = 0,
    emit_every: int = 50,
) -> dict[str, object]:
    payload = build_tsp_arena_bundle(
        tsplib_dir=tsplib_dir,
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
