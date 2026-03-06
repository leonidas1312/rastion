from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from rastion.plugins.discovery import discover_solvers
from rastion.solvers.registry import get_solver
from rastion.tsp.tsplib import load_tsplib_problem

from .loaders import DEFAULT_CARDS_DIR, DEFAULT_SUITES_DIR, REPO_ROOT, load_solver_cards, load_suite_specs
from .schema import EvalRunRecord, EvalSuiteSpec, ListingTier, SolverCard


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate_cards(
    cards: list[SolverCard],
    suite: EvalSuiteSpec,
) -> list[SolverCard]:
    allowed_tiers = set(suite.listing_tiers)
    out: list[SolverCard] = []
    for card in cards:
        if suite.problem_variant not in card.problem_variants:
            continue
        if card.listing_tier not in allowed_tiers:
            continue
        out.append(card)
    return sorted(out, key=lambda item: (item.listing_tier, item.name))


def _load_problem(problem_variant: str, path: str):
    if problem_variant == "tsp":
        return load_tsplib_problem(REPO_ROOT / path)
    raise ValueError(f"Unsupported problem variant: {problem_variant}")


def _record_status(exc: Exception) -> str:
    message = str(exc).lower()
    if "not installed" in message or "import" in message:
        return "dependency-missing"
    return "error"


def evaluate_suite(
    suite_id: str,
    *,
    cards_dir: str | Path = DEFAULT_CARDS_DIR,
    suites_dir: str | Path = DEFAULT_SUITES_DIR,
) -> dict[str, object]:
    discover_solvers(plugin_root=REPO_ROOT / "plugins_local", use_entry_points=False, reset_registry=True)

    cards = [loaded.card for loaded in load_solver_cards(cards_dir)]
    suites = {suite.id: suite for suite in load_suite_specs(suites_dir)}
    suite = suites[suite_id]
    selected = _candidate_cards(cards, suite)

    suite_results: list[dict[str, object]] = []
    standings: dict[str, dict[str, object]] = {}
    generated_at = _now_utc_iso()

    for instance in suite.instances:
        problem = _load_problem(suite.problem_variant, instance.path)
        instance_results: list[dict[str, object]] = []

        for card in selected:
            solver = get_solver(card.adapter.solver_name)
            scores: list[float] = []
            runtimes: list[float] = []
            record_status = "ok"
            error_message: str | None = None

            for seed in suite.seed_policy.seeds:
                kwargs: dict[str, int] = {"seed": seed}
                if suite.budget_policy.iters is not None:
                    kwargs["iters"] = int(suite.budget_policy.iters)
                if suite.budget_policy.time_budget_ms is not None:
                    kwargs["time_budget_ms"] = int(suite.budget_policy.time_budget_ms)

                started = time.perf_counter()
                try:
                    solution = solver.solve(problem, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    record_status = _record_status(exc)
                    error_message = str(exc)
                    break

                runtimes.append(max((time.perf_counter() - started) * 1000.0, 0.0))
                scores.append(float(solution.best_value))

            score = min(scores) if scores else None
            avg_runtime_ms = (sum(runtimes) / len(runtimes)) if runtimes else None
            run_record = EvalRunRecord(
                suite_id=suite.id,
                solver_id=card.id,
                solver_version=card.version,
                problem_variant=suite.problem_variant,
                instance_id=instance.id,
                score=score,
                runtime=avg_runtime_ms,
                status=record_status,
                seed_policy=suite.seed_policy.model_dump(),
                budget_policy=suite.budget_policy.model_dump(),
                generated_at=generated_at,
                artifact_path=f"data/evals/{suite.id}.json",
                environment={
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                },
            )

            result_row = {
                "solver_id": card.id,
                "solver_name": card.name,
                "solver_slug": card.id,
                "solver_version": card.version,
                "method_class": card.method_class,
                "listing_tier": card.listing_tier,
                "score": score,
                "avg_runtime_ms": avg_runtime_ms,
                "status": record_status,
                "error": error_message,
                "run_record": run_record.model_dump(),
            }
            instance_results.append(result_row)

            standing = standings.setdefault(
                card.id,
                {
                    "solver_id": card.id,
                    "solver_name": card.name,
                    "solver_slug": card.id,
                    "method_class": card.method_class,
                    "listing_tier": card.listing_tier,
                    "scores": [],
                    "runtimes": [],
                    "status_counts": {},
                },
            )
            standing["status_counts"][record_status] = standing["status_counts"].get(record_status, 0) + 1
            if score is not None:
                standing["scores"].append(score)
            if avg_runtime_ms is not None:
                standing["runtimes"].append(avg_runtime_ms)

        instance_results.sort(
            key=lambda row: (
                row["status"] != "ok",
                float("inf") if row["score"] is None else float(row["score"]),
                float("inf") if row["avg_runtime_ms"] is None else float(row["avg_runtime_ms"]),
            )
        )
        suite_results.append(
            {
                "id": instance.id,
                "label": instance.label,
                "size": instance.size,
                "path": instance.path,
                "n_vars": int(problem.n_vars),
                "results": instance_results,
            }
        )

    standings_rows: list[dict[str, object]] = []
    for standing in standings.values():
        scores = standing.pop("scores")
        runtimes = standing.pop("runtimes")
        standings_rows.append(
            {
                **standing,
                "mean_score": (sum(scores) / len(scores)) if scores else None,
                "mean_runtime_ms": (sum(runtimes) / len(runtimes)) if runtimes else None,
                "completed_instances": len(scores),
                "total_instances": len(suite.instances),
            }
        )

    standings_rows.sort(
        key=lambda row: (
            row["completed_instances"] == 0,
            float("inf") if row["mean_score"] is None else float(row["mean_score"]),
            float("inf") if row["mean_runtime_ms"] is None else float(row["mean_runtime_ms"]),
        )
    )

    return {
        "generated_at": generated_at,
        "suite": suite.model_dump(),
        "instances": suite_results,
        "standings": standings_rows,
    }


def write_suite_eval_json(
    out_path: str | Path,
    suite_id: str,
    *,
    cards_dir: str | Path = DEFAULT_CARDS_DIR,
    suites_dir: str | Path = DEFAULT_SUITES_DIR,
) -> dict[str, object]:
    payload = evaluate_suite(suite_id, cards_dir=cards_dir, suites_dir=suites_dir)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def evaluate_all_suites(
    *,
    out_dir: str | Path | None = None,
    cards_dir: str | Path = DEFAULT_CARDS_DIR,
    suites_dir: str | Path = DEFAULT_SUITES_DIR,
) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for suite in load_suite_specs(suites_dir):
        payload = evaluate_suite(suite.id, cards_dir=cards_dir, suites_dir=suites_dir)
        if out_dir is not None:
            out_path = Path(out_dir) / f"{suite.id}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payloads.append(payload)
    return payloads
