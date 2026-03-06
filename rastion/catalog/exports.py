from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rastion.plugins.discovery import discover_solvers
from rastion.solvers.registry import list_solvers

from .evals import evaluate_all_suites
from .loaders import DEFAULT_CARDS_DIR, DEFAULT_SUITES_DIR, REPO_ROOT, load_solver_cards, load_suite_specs


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def export_catalog_json(
    out_path: str | Path,
    *,
    cards_dir: str | Path = DEFAULT_CARDS_DIR,
    suites_dir: str | Path = DEFAULT_SUITES_DIR,
) -> dict[str, object]:
    discover_solvers(plugin_root=REPO_ROOT / "plugins_local", use_entry_points=False, reset_registry=True)
    runtime = {row.name: row for row in list_solvers()}
    suites = load_suite_specs(suites_dir)

    rows: list[dict[str, object]] = []
    for loaded in load_solver_cards(cards_dir):
        card = loaded.card
        runtime_row = runtime.get(card.adapter.solver_name)
        rows.append(
            {
                **card.model_dump(mode="json"),
                "detail_path": f"solvers/{card.id}/",
                "official_suite_ids": [
                    suite.id for suite in suites if suite.problem_variant in card.problem_variants and card.listing_tier in suite.listing_tiers
                ],
                "runtime": None
                if runtime_row is None
                else {
                    "solver_name": runtime_row.name,
                    "supports": list(runtime_row.solver.supports),
                    "max_size": int(runtime_row.solver.max_size),
                    "hardware": list(runtime_row.solver.hardware),
                    "quality": float(runtime_row.metadata.quality),
                    "source": runtime_row.metadata.source,
                },
            }
        )

    rows.sort(key=lambda item: (str(item["listing_tier"]), str(item["name"])))
    payload = {
        "generated_at": _now_utc_iso(),
        "family": "routing",
        "problem_variants": ["tsp", "cvrp", "vrptw", "pickup-delivery", "other"],
        "count": len(rows),
        "solvers": rows,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def export_suites_json(
    out_path: str | Path,
    *,
    suites_dir: str | Path = DEFAULT_SUITES_DIR,
) -> dict[str, object]:
    suites = [suite.model_dump() for suite in load_suite_specs(suites_dir)]
    payload = {
        "generated_at": _now_utc_iso(),
        "count": len(suites),
        "suites": suites,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def export_leaderboards_json(
    out_path: str | Path,
    *,
    suite_payloads: list[dict[str, object]] | None = None,
    evals_out_dir: str | Path | None = None,
    cards_dir: str | Path = DEFAULT_CARDS_DIR,
    suites_dir: str | Path = DEFAULT_SUITES_DIR,
) -> dict[str, object]:
    payloads = suite_payloads or evaluate_all_suites(
        out_dir=evals_out_dir,
        cards_dir=cards_dir,
        suites_dir=suites_dir,
    )
    rows = []
    for suite_payload in payloads:
        suite = suite_payload.get("suite", {})
        rows.append(
            {
                "suite": suite,
                "generated_at": suite_payload.get("generated_at"),
                "standings": suite_payload.get("standings", []),
                "instances": suite_payload.get("instances", []),
                "artifact_path": f"data/evals/{suite.get('id', 'unknown')}.json",
            }
        )

    rows.sort(key=lambda item: str(item["suite"].get("title", "")))
    payload = {
        "generated_at": _now_utc_iso(),
        "count": len(rows),
        "leaderboards": rows,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
