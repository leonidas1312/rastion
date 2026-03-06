from __future__ import annotations

import json
from pathlib import Path

import pytest

from rastion.catalog.evals import evaluate_suite
from rastion.catalog.exports import export_catalog_json, export_leaderboards_json, export_suites_json
from rastion.catalog.loaders import validate_catalog


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_validate_catalog_accepts_repo_cards_and_suites() -> None:
    cards, suites = validate_catalog()

    assert len(cards) >= 3
    assert len(suites) == 3
    assert {loaded.card.id for loaded in cards} >= {
        "tsp-nearest-neighbor",
        "tsp-two-opt",
        "tsp-ortools",
    }


def test_validate_catalog_rejects_missing_detail_markdown(tmp_path: Path) -> None:
    cards_dir = tmp_path / "cards"
    suites_dir = tmp_path / "suites"
    solver_dir = cards_dir / "broken-solver"
    solver_dir.mkdir(parents=True)
    suites_dir.mkdir(parents=True)

    card = {
        "id": "broken-solver",
        "name": "Broken Solver",
        "routing_family": "routing",
        "problem_variants": ["tsp"],
        "summary": "broken",
        "description": "broken",
        "repository": "https://github.com/leonidas1312/rastion",
        "homepage": "https://github.com/leonidas1312/rastion",
        "license": "Apache-2.0",
        "authors": ["rastion contributors"],
        "maintainers": ["rastion contributors"],
        "installation": {
            "command": "pip install -e ."
        },
        "adapter": {
            "kind": "local_plugin",
            "solver_name": "tsp_nearest_neighbor",
            "path": "plugins_local/rastion_tsp_nearest",
            "entrypoint": None
        },
        "capabilities": {
            "streaming": True,
            "deterministic_mode": True,
            "warm_start": False,
            "local_search": False,
            "multi_route_ready": False
        },
        "limits": {
            "tested_max_nodes": 10,
            "unsupported_constraints": [],
            "depot_assumptions": "single depot"
        },
        "hardware": ["cpu"],
        "determinism": "fixed seed",
        "method_class": "heuristic",
        "version": "0.1.0",
        "citations": [],
        "references": [],
        "known_failure_modes": [],
        "artifact_paths": {
            "benchmark_json": "data/benchmarks.json",
            "arena_json": "data/tsp_arena.json",
            "suite_results_dir": "data/evals/",
            "detail_markdown": "catalog/solvers/broken-solver/README.md"
        },
        "tags": ["tsp"],
        "listing_tier": "official"
    }
    suite = {
        "id": "suite",
        "title": "Suite",
        "description": "suite",
        "problem_variant": "tsp",
        "instances": [
            {
                "id": "berlin52",
                "label": "Berlin52",
                "path": "examples/tsplib/berlin52.tsp",
                "size": "small"
            }
        ],
        "budget_policy": {
            "iters": 10,
            "time_budget_ms": 100,
            "repeat": 1
        },
        "seed_policy": {
            "seeds": [0]
        },
        "metric_policy": {
            "primary_metric": "distance",
            "objective": "minimize",
            "runtime_unit": "ms"
        },
        "result_schema_version": "1.0",
        "optional_dependencies": [],
        "listing_tiers": ["official"]
    }

    (solver_dir / "card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    (suites_dir / "suite.json").write_text(json.dumps(suite, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="detail markdown missing"):
        validate_catalog(cards_dir=cards_dir, suites_dir=suites_dir)


def test_export_catalog_and_suites_write_routing_payloads(tmp_path: Path) -> None:
    catalog_out = tmp_path / "catalog.json"
    suites_out = tmp_path / "suites.json"

    catalog_payload = export_catalog_json(catalog_out)
    suites_payload = export_suites_json(suites_out)

    assert catalog_out.exists()
    assert suites_out.exists()
    assert catalog_payload["family"] == "routing"
    assert "tsp" in catalog_payload["problem_variants"]
    assert suites_payload["count"] == 3


def test_evaluate_suite_and_export_leaderboards_emit_tsp_results(tmp_path: Path) -> None:
    evals_dir = tmp_path / "evals"
    evals_dir.mkdir(parents=True)

    suite_payload = evaluate_suite("tsplib-small-v1")
    leaders_out = tmp_path / "leaderboards.json"
    leaders_payload = export_leaderboards_json(leaders_out, suite_payloads=[suite_payload], evals_out_dir=evals_dir)

    assert leaders_out.exists()
    assert leaders_payload["count"] == 1

    standings = leaders_payload["leaderboards"][0]["standings"]
    solver_ids = {row["solver_id"] for row in standings}
    assert "tsp-nearest-neighbor" in solver_ids
    assert "tsp-two-opt" in solver_ids

    by_id = {row["solver_id"]: row for row in standings}
    assert by_id["tsp-nearest-neighbor"]["mean_score"] is not None
    assert by_id["tsp-two-opt"]["mean_score"] is not None
