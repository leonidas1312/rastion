from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rastion.plugins.discovery import discover_solvers
from rastion.solvers.registry import list_solvers

from .schema import EvalSuiteSpec, SolverCard


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CARDS_DIR = REPO_ROOT / "catalog" / "solvers"
DEFAULT_SUITES_DIR = REPO_ROOT / "catalog" / "suites"


@dataclass(slots=True)
class LoadedSolverCard:
    card: SolverCard
    card_path: Path
    solver_dir: Path


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_solver_cards(cards_dir: str | Path = DEFAULT_CARDS_DIR) -> list[LoadedSolverCard]:
    root = Path(cards_dir)
    loaded: list[LoadedSolverCard] = []

    if not root.exists():
        return loaded

    for card_path in sorted(root.glob("*/card.json")):
        solver_dir = card_path.parent
        loaded.append(
            LoadedSolverCard(
                card=SolverCard.model_validate(_load_json(card_path)),
                card_path=card_path,
                solver_dir=solver_dir,
            )
        )

    return loaded


def load_suite_specs(suites_dir: str | Path = DEFAULT_SUITES_DIR) -> list[EvalSuiteSpec]:
    root = Path(suites_dir)
    loaded: list[EvalSuiteSpec] = []

    if not root.exists():
        return loaded

    for suite_path in sorted(root.glob("*.json")):
        loaded.append(EvalSuiteSpec.model_validate(_load_json(suite_path)))

    return loaded


def validate_catalog(
    *,
    cards_dir: str | Path = DEFAULT_CARDS_DIR,
    suites_dir: str | Path = DEFAULT_SUITES_DIR,
    discover_runtime_solvers: bool = True,
) -> tuple[list[LoadedSolverCard], list[EvalSuiteSpec]]:
    loaded_cards = load_solver_cards(cards_dir)
    suite_specs = load_suite_specs(suites_dir)
    issues: list[str] = []

    if not loaded_cards:
        issues.append(f"No solver cards found in {Path(cards_dir)}")
    if not suite_specs:
        issues.append(f"No suite specs found in {Path(suites_dir)}")

    card_ids: set[str] = set()
    for loaded in loaded_cards:
        card = loaded.card
        if card.id in card_ids:
            issues.append(f"Duplicate solver card id: {card.id}")
        card_ids.add(card.id)

        if card.routing_family != "routing":
            issues.append(f"{card.id}: routing_family must be 'routing'")
        if "tsp" not in card.problem_variants:
            issues.append(f"{card.id}: Phase 1 cards must declare tsp support")

        adapter_path = REPO_ROOT / card.adapter.path
        if not adapter_path.exists():
            issues.append(f"{card.id}: adapter path does not exist: {card.adapter.path}")

        detail_path = REPO_ROOT / card.artifact_paths.detail_markdown
        if not detail_path.exists():
            issues.append(f"{card.id}: detail markdown missing: {card.artifact_paths.detail_markdown}")

    suite_ids: set[str] = set()
    for suite in suite_specs:
        if suite.id in suite_ids:
            issues.append(f"Duplicate suite id: {suite.id}")
        suite_ids.add(suite.id)

        if suite.problem_variant != "tsp":
            issues.append(f"{suite.id}: Phase 1 suites must target tsp")

        for instance in suite.instances:
            instance_path = REPO_ROOT / instance.path
            if not instance_path.exists():
                issues.append(f"{suite.id}: missing instance path {instance.path}")

    if discover_runtime_solvers:
        discover_solvers(plugin_root=REPO_ROOT / "plugins_local", use_entry_points=False, reset_registry=True)
        runtime = {row.name: row for row in list_solvers()}
        for loaded in loaded_cards:
            card = loaded.card
            row = runtime.get(card.adapter.solver_name)
            if row is None:
                issues.append(f"{card.id}: solver '{card.adapter.solver_name}' is not discoverable")
                continue
            if "tsp" not in row.solver.supports:
                issues.append(f"{card.id}: solver '{card.adapter.solver_name}' does not support tsp")

    if issues:
        raise ValueError("\n".join(issues))

    return loaded_cards, suite_specs
