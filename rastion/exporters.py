from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rastion.benchmark import DEFAULT_BENCHMARK_PATH
from rastion.catalog.exports import export_catalog_json, export_leaderboards_json, export_suites_json
from rastion.solvers.registry import list_solvers


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def export_solvers_json(out_path: str | Path) -> dict[str, object]:
    rows = []
    for row in list_solvers():
        metadata = row.metadata.model_dump()
        rows.append(
            {
                "name": row.name,
                "supports": metadata.get("supports", []),
                "max_size": int(metadata.get("max_size", row.solver.max_size)),
                "hardware": metadata.get("hardware", []),
                "quality": float(metadata.get("quality", row.solver.quality)),
                "source": metadata.get("source", "unknown"),
                "version": metadata.get("version"),
            }
        )

    rows.sort(key=lambda item: item["name"])
    payload = {
        "generated_at": _now_utc_iso(),
        "scope": "developer",
        "note": "Developer export only. The public website uses catalog, suites, leaderboards, evals, and tsp_arena artifacts.",
        "count": len(rows),
        "solvers": rows,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _parse_benchmark_key(key: str) -> tuple[str, int, str]:
    parts = key.split("|", maxsplit=2)
    if len(parts) != 3:
        raise ValueError(f"Invalid benchmark key: {key}")
    return parts[0], int(parts[1]), parts[2]


def export_benchmarks_json(
    out_path: str | Path,
    *,
    cache_path: str | Path = DEFAULT_BENCHMARK_PATH,
) -> dict[str, object]:
    cache = Path(cache_path)
    if cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
    else:
        payload = {"entries": {}}

    entries = payload.get("entries", {}) if isinstance(payload, dict) else {}

    rows: list[dict[str, object]] = []
    for key, value in entries.items():
        if not isinstance(value, dict):
            continue
        try:
            problem_type, n_vars, solver_name = _parse_benchmark_key(str(key))
        except ValueError:
            continue

        rows.append(
            {
                "problem_type": problem_type,
                "n_vars": n_vars,
                "solver_name": solver_name,
                "hint": float(value.get("hint", 0.0)),
                "best_value": float(value.get("best_value", 0.0)),
                "avg_runtime": float(value.get("avg_runtime", 0.0)),
                "updated_at": str(value.get("updated_at", "")),
            }
        )

    rows.sort(key=lambda item: (str(item["problem_type"]), int(item["n_vars"]), str(item["solver_name"])))

    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["problem_type"]), int(row["n_vars"]))
        grouped.setdefault(key, []).append(row)

    problems = [
        {
            "problem_type": key[0],
            "n_vars": key[1],
            "results": sorted(value, key=lambda item: (float(item["best_value"]), float(item["avg_runtime"]))),
        }
        for key, value in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))
    ]

    normalized = {
        "generated_at": _now_utc_iso(),
        "scope": "developer",
        "note": "Developer export only. The public website does not use benchmark-cache artifacts.",
        "source_cache": str(cache),
        "rows": rows,
        "problems": problems,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return normalized


__all__ = [
    "export_benchmarks_json",
    "export_catalog_json",
    "export_leaderboards_json",
    "export_solvers_json",
    "export_suites_json",
]
