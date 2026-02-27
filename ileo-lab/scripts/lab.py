#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
WORKSPACE_ROOT = LAB_ROOT / "workspace"
PROBLEMS_ROOT = WORKSPACE_ROOT / "problems"
ARCHIVES_ROOT = WORKSPACE_ROOT / "archives"
EXPORTS_ROOT = WORKSPACE_ROOT / "exports"
SOLVER_PACKAGES_ROOT = LAB_ROOT / "packages" / "solvers"
RASTION_HOME = LAB_ROOT / ".rastion-home"

DEFAULT_PROBLEM_NAMES = ["lab_knapsack_stress", "lab_maxcut_stress"]
DEFAULT_SOLVER_NAMES = ["grasp_binary_pro", "qubo_tabu_pro"]


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "create":
        cmd_create(args)
        return 0
    if args.command == "load":
        cmd_load(args)
        return 0
    if args.command == "test":
        cmd_test(args)
        return 0
    if args.command == "hub-roundtrip":
        cmd_hub_roundtrip(args)
        return 0
    if args.command == "all":
        cmd_create(args)
        cmd_load(args)
        cmd_test(args)
        return 0

    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ileo-lab",
        description="Fresh lab workflow for testing rastion CLI/TUI/core plus heuristic packages.",
    )
    sub = parser.add_subparsers(dest="command")

    create = sub.add_parser("create", help="Generate stress problems and build solver archives")
    create.add_argument("--clean", action="store_true", help="Delete existing workspace before regeneration")

    load = sub.add_parser("load", help="Install generated packages into isolated RASTION_HOME")
    load.add_argument("--force-create", action="store_true", help="Always recreate artifacts before installing")

    test = sub.add_parser("test", help="Run validation/solve/benchmark/export smoke tests")
    test.add_argument("--with-pytests", action="store_true", help="Also run selected repository pytest checks")

    hub = sub.add_parser("hub-roundtrip", help="Push and pull one problem and solver through hub")
    hub.add_argument("--hub-url", type=str, default=None, help="Hub base URL (defaults to config value)")
    hub.add_argument("--github-token", type=str, default=None, help="GitHub token for login exchange")
    hub.add_argument("--problem", type=str, default="lab_knapsack_stress", help="Generated problem package name")
    hub.add_argument("--solver", type=str, default="grasp_binary_pro", help="Local solver package folder name")

    sub.add_parser("all", help="Run create + load + test sequence")
    return parser


def cmd_create(args: argparse.Namespace) -> None:
    if getattr(args, "clean", False) and WORKSPACE_ROOT.exists():
        shutil.rmtree(WORKSPACE_ROOT)

    PROBLEMS_ROOT.mkdir(parents=True, exist_ok=True)
    ARCHIVES_ROOT.mkdir(parents=True, exist_ok=True)
    EXPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    create_knapsack_problem(PROBLEMS_ROOT / "lab_knapsack_stress", n_items=96, seed=17)
    create_maxcut_problem(PROBLEMS_ROOT / "lab_maxcut_stress", n_nodes=64, seed=23)
    build_solver_archives()

    print("Created problem packages:")
    for path in sorted(PROBLEMS_ROOT.iterdir()):
        if path.is_dir():
            print(f"- {path}")

    print("Built solver archives:")
    for archive in sorted(ARCHIVES_ROOT.glob("*.zip")):
        print(f"- {archive}")


def cmd_load(args: argparse.Namespace) -> None:
    if getattr(args, "force_create", False) or not _artifacts_exist():
        cmd_create(argparse.Namespace(clean=False))

    env = base_env()
    run_cli(["init-registry", "--force"], env=env)

    for problem_dir in sorted(PROBLEMS_ROOT.iterdir()):
        if not problem_dir.is_dir():
            continue
        run_cli(["install", str(problem_dir), "--overwrite"], env=env)

    for archive in sorted(ARCHIVES_ROOT.glob("*.zip")):
        run_cli(["install-solver", archive.resolve().as_uri(), "--overwrite"], env=env)

    run_cli(["info"], env=env)
    run_cli(["info", "--verbose", "lab_knapsack_stress", "default"], env=env)


def cmd_test(args: argparse.Namespace) -> None:
    if not _artifacts_exist():
        cmd_create(argparse.Namespace(clean=False))

    # Ensure generated packages are installed into the isolated registry before testing.
    cmd_load(argparse.Namespace(force_create=False))
    env = base_env()
    run_cli(["validate", "lab_knapsack_stress", "default"], env=env)
    run_cli(["validate", "lab_knapsack_stress", "stress"], env=env)
    run_cli(["validate", "lab_maxcut_stress", "default"], env=env)
    run_cli(["validate", "lab_maxcut_stress", "stress"], env=env)

    run_cli(
        [
            "solve",
            "lab_knapsack_stress",
            "default",
            "--solver",
            "grasp_binary_pro",
            "--time-limit",
            "3",
            "--seed",
            "7",
        ],
        env=env,
    )
    run_cli(
        [
            "solve",
            "lab_maxcut_stress",
            "default",
            "--solver",
            "qubo_tabu_pro",
            "--time-limit",
            "3",
            "--seed",
            "7",
        ],
        env=env,
    )
    run_cli(
        [
            "benchmark",
            "lab_knapsack_stress",
            "--instance",
            "default",
            "--solvers",
            "grasp_binary_pro",
            "baseline",
            "--runs",
            "2",
            "--time-limit",
            "2s",
        ],
        env=env,
    )
    run_cli(
        [
            "benchmark",
            "lab_maxcut_stress",
            "--instance",
            "default",
            "--solvers",
            "qubo_tabu_pro",
            "--runs",
            "2",
            "--time-limit",
            "2s",
        ],
        env=env,
    )

    export_generated_problems(env)
    tui_smoke(env)

    if getattr(args, "with_pytests", False):
        run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_validation.py",
                "tests/test_registry_api.py",
                "-q",
            ],
            env=env,
            cwd=REPO_ROOT,
        )


def cmd_hub_roundtrip(args: argparse.Namespace) -> None:
    if not _artifacts_exist():
        cmd_create(argparse.Namespace(clean=False))

    problem_path = PROBLEMS_ROOT / args.problem
    solver_path = SOLVER_PACKAGES_ROOT / args.solver

    if not problem_path.exists():
        raise FileNotFoundError(f"problem package not found: {problem_path}")
    if not solver_path.exists():
        raise FileNotFoundError(f"solver package not found: {solver_path}")

    os.environ.update(base_env())

    async def _run_roundtrip() -> dict[str, Any]:
        from rastion.hub.client import HubClient

        client = HubClient(base_url=args.hub_url)
        if args.github_token:
            user = await client.login(args.github_token)
            print(f"Authenticated as {user.get('username', 'unknown')}")
        elif not client.token:
            raise RuntimeError("No hub auth token found. Provide --github-token or login first.")

        pushed_problem = await client.push(problem_path, type="problem")
        pushed_solver = await client.push(solver_path, type="solver")

        pulled_problem = await client.pull(str(pushed_problem.get("name") or args.problem), type="problem", overwrite=True)
        pulled_solver = await client.pull(str(pushed_solver.get("name") or args.solver), type="solver", overwrite=True)

        return {
            "pushed_problem": pushed_problem,
            "pulled_problem": pulled_problem,
            "pushed_solver": pushed_solver,
            "pulled_solver": pulled_solver,
        }

    summary = asyncio.run(_run_roundtrip())
    print(json.dumps(summary, indent=2, sort_keys=True))


def create_knapsack_problem(root: Path, *, n_items: int, seed: int) -> None:
    rng = random.Random(seed)
    instances_dir = root / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    variables = [{"name": f"x{i}", "vartype": "binary"} for i in range(n_items)]
    profits = [rng.randint(8, 120) for _ in range(n_items)]
    weights = [rng.randint(2, 45) for _ in range(n_items)]
    capacity = int(sum(weights) * 0.33)

    profits_stress = [int(p * rng.uniform(0.8, 1.2)) for p in profits]
    weights_stress = [max(1, int(w * rng.uniform(0.8, 1.25))) for w in weights]
    capacity_stress = int(sum(weights_stress) * 0.3)

    spec = {
        "schema_version": "0.1.0",
        "name": "lab_knapsack_stress",
        "ir_target": "generic",
        "variables": variables,
        "objective": {
            "sense": "max",
            "linear": "profits",
            "constant": 0.0,
        },
        "constraints": [
            {
                "name": "capacity",
                "matrix": "A_capacity",
                "rhs": "b_capacity",
                "sense": "<=",
            }
        ],
    }

    default_instance = {
        "schema_version": "0.1.0",
        "arrays": {
            "profits": profits,
            "A_capacity": [weights],
            "b_capacity": [capacity],
        },
        "params": {
            "notes": "Generated knapsack stress instance",
            "seed": seed,
            "n_items": n_items,
        },
    }

    stress_instance = {
        "schema_version": "0.1.0",
        "arrays": {
            "profits": profits_stress,
            "A_capacity": [weights_stress],
            "b_capacity": [capacity_stress],
        },
        "params": {
            "notes": "Harder capacity ratio",
            "seed": seed + 1,
            "n_items": n_items,
        },
    }

    metadata = {
        "name": "lab_knapsack_stress",
        "version": "0.1.0",
        "author": "ileo-lab",
        "description": "Generated binary knapsack benchmark for heuristic testing",
        "optimization_class": "MILP",
        "difficulty": "medium",
        "tags": ["lab", "knapsack", "binary", "heuristic"],
    }

    card = """# lab_knapsack_stress

Generated knapsack package for CLI/TUI/core stress tests.

- 96 binary decision variables
- single capacity constraint
- `default` and `stress` instances
- suitable for `baseline` and `grasp_binary_pro`
"""

    write_json(root / "spec.json", spec)
    write_json(instances_dir / "default.json", default_instance)
    write_json(instances_dir / "stress.json", stress_instance)
    write_metadata(root / "metadata.yaml", metadata)
    (root / "problem_card.md").write_text(card.strip() + "\n", encoding="utf-8")


def create_maxcut_problem(root: Path, *, n_nodes: int, seed: int) -> None:
    rng = random.Random(seed)
    instances_dir = root / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    variables = [{"name": f"x{i}", "vartype": "binary"} for i in range(n_nodes)]

    def graph_to_qubo(edges: list[tuple[int, int, float]]) -> tuple[list[float], list[list[float]]]:
        linear = [0.0] * n_nodes
        q = [[0.0 for _ in range(n_nodes)] for _ in range(n_nodes)]
        for i, j, weight in edges:
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            linear[a] -= weight
            linear[b] -= weight
            q[a][b] += 2.0 * weight
        return linear, q

    def build_edges(density: float) -> list[tuple[int, int, float]]:
        edges: list[tuple[int, int, float]] = []
        for i in range(n_nodes):
            j = (i + 1) % n_nodes
            weight = float(rng.randint(1, 6))
            edges.append((i, j, weight))
        for i in range(n_nodes):
            for j in range(i + 2, n_nodes):
                if rng.random() <= density:
                    weight = float(rng.randint(1, 9))
                    edges.append((i, j, weight))
        return edges

    default_edges = build_edges(0.12)
    stress_edges = build_edges(0.22)

    linear_default, q_default = graph_to_qubo(default_edges)
    linear_stress, q_stress = graph_to_qubo(stress_edges)

    spec = {
        "schema_version": "0.1.0",
        "name": "lab_maxcut_stress",
        "ir_target": "qubo",
        "variables": variables,
        "objective": {
            "sense": "min",
            "linear": "linear",
            "quadratic": "Q",
            "constant": 0.0,
        },
        "constraints": [],
    }

    default_instance = {
        "schema_version": "0.1.0",
        "arrays": {
            "linear": linear_default,
            "Q": q_default,
        },
        "params": {
            "notes": "Generated MaxCut-to-QUBO instance",
            "seed": seed,
            "n_nodes": n_nodes,
            "edge_count": len(default_edges),
        },
    }

    stress_instance = {
        "schema_version": "0.1.0",
        "arrays": {
            "linear": linear_stress,
            "Q": q_stress,
        },
        "params": {
            "notes": "Denser graph variant",
            "seed": seed + 1,
            "n_nodes": n_nodes,
            "edge_count": len(stress_edges),
        },
    }

    metadata = {
        "name": "lab_maxcut_stress",
        "version": "0.1.0",
        "author": "ileo-lab",
        "description": "Generated MaxCut QUBO benchmark for heuristic sampler tests",
        "optimization_class": "QUBO",
        "difficulty": "medium",
        "tags": ["lab", "maxcut", "qubo", "heuristic"],
    }

    card = """# lab_maxcut_stress

Generated MaxCut-style QUBO package for heuristic sampler evaluation.

- 64 binary variables
- no linear constraints
- weighted graph converted into QUBO energy
- `default` and `stress` density variants
"""

    write_json(root / "spec.json", spec)
    write_json(instances_dir / "default.json", default_instance)
    write_json(instances_dir / "stress.json", stress_instance)
    write_metadata(root / "metadata.yaml", metadata)
    (root / "problem_card.md").write_text(card.strip() + "\n", encoding="utf-8")


def build_solver_archives() -> None:
    ARCHIVES_ROOT.mkdir(parents=True, exist_ok=True)
    for solver_dir in sorted(SOLVER_PACKAGES_ROOT.iterdir()):
        if not solver_dir.is_dir():
            continue
        solver_file = solver_dir / "solver.py"
        if not solver_file.exists():
            continue

        archive_path = ARCHIVES_ROOT / f"{solver_dir.name}.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for filename in ["solver.py", "metadata.yaml", "solver_card.md", "README.md"]:
                source = solver_dir / filename
                if source.exists() and source.is_file():
                    zf.write(source, arcname=f"{solver_dir.name}/{filename}")


def export_generated_problems(env: dict[str, str]) -> None:
    EXPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    for problem_name in DEFAULT_PROBLEM_NAMES:
        target = EXPORTS_ROOT / problem_name
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        run_cli(["export", problem_name, str(target)], env=env)


def tui_smoke(env: dict[str, str]) -> None:
    if importlib.util.find_spec("readchar") is None:
        print("Skipping TUI import smoke: optional dependency 'readchar' is not installed.")
        return

    run(
        [
            sys.executable,
            "-c",
            "import rastion.tui.menu as menu; print('tui import smoke: ok')",
        ],
        env=env,
        cwd=REPO_ROOT,
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    for key, value in payload.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def base_env() -> dict[str, str]:
    env = dict(os.environ)
    env["RASTION_HOME"] = str(RASTION_HOME)

    py_path = env.get("PYTHONPATH", "")
    if py_path:
        env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{py_path}"
    else:
        env["PYTHONPATH"] = str(REPO_ROOT)

    return env


def run_cli(args: list[str], *, env: dict[str, str]) -> None:
    run([sys.executable, "-m", "rastion.cli.main", *args], env=env, cwd=REPO_ROOT)


def run(command: list[str], *, env: dict[str, str], cwd: Path) -> None:
    print(f"$ {shlex.join(command)}")
    subprocess.run(command, env=env, cwd=cwd, check=True)


def _artifacts_exist() -> bool:
    if not PROBLEMS_ROOT.exists() or not ARCHIVES_ROOT.exists():
        return False

    required_problem_dirs = all((PROBLEMS_ROOT / name).exists() for name in DEFAULT_PROBLEM_NAMES)
    required_solver_archives = all((ARCHIVES_ROOT / f"{name}.zip").exists() for name in DEFAULT_SOLVER_NAMES)
    return required_problem_dirs and required_solver_archives


if __name__ == "__main__":
    raise SystemExit(main())
