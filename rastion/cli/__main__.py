"""Command line interface for rastion."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any

from rastion.agent import init_agent_workspace, run_agent_request_file
from rastion.backends.local import LocalBackend
from rastion.benchmark import compare
from rastion.compile.normalize import compile_to_ir
from rastion.config import get_token
from rastion.core.data import InstanceData
from rastion.core.run_record import append_run_record, create_run_record
from rastion.core.solution import Solution, SolutionStatus
from rastion.core.spec import ProblemSpec
from rastion.core.validate import validate_problem_and_instance
from rastion.hub import HubClient, HubClientError
from rastion.registry.loader import DecisionPlugin
from rastion.registry.manager import (
    export_decision_plugin,
    init_registry,
    install_decision_plugin,
    install_solver_from_url,
    list_decision_plugins,
    problems_root,
    read_yaml_file,
    remove_decision_plugin,
    runs_file,
    runs_root,
    solver_hub_source,
)
from rastion.solvers.discovery import discover_plugins
from rastion.solvers.matching import (
    auto_select_solver,
    compatible_plugins,
    match_report,
    required_capabilities_summary,
    requirements_from_ir,
)
from rastion.version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rastion",
        description="Local decision-plugin optimization runtime.",
    )
    parser.add_argument("--version", action="version", version=f"rastion {__version__}")

    sub = parser.add_subparsers(dest="command")

    info = sub.add_parser("info", help="Show installed solver plugins and capabilities")
    info.add_argument("--verbose", action="store_true", help="Show capabilities and compatibility details")
    info.add_argument("spec", nargs="?", type=str, help="Decision plugin spec path OR registry plugin name")
    info.add_argument("instance", nargs="?", type=str, help="Instance path OR registry instance name")
    info.set_defaults(func=cmd_info)

    validate = sub.add_parser("validate", help="Validate decision plugin spec + instance data")
    validate.add_argument("spec", type=str, help="Path to spec.json OR registry plugin name")
    validate.add_argument("instance", type=str, help="Path to instance JSON/NPZ OR registry instance name")
    validate.set_defaults(func=cmd_validate)

    solve = sub.add_parser("solve", help="Solve a decision plugin instance")
    solve.add_argument("spec", type=str, help="Path to spec.json OR registry plugin name")
    solve.add_argument("instance", type=str, help="Path to instance JSON/NPZ OR registry instance name")
    solve.add_argument("--solver", default="auto", help="Solver name or auto")
    solve.add_argument(
        "--preferred",
        action="append",
        default=[],
        help="Preferred solver in auto mode; can be repeated",
    )
    solve.add_argument("--time-limit", type=float, default=None, help="Time limit in seconds")
    solve.add_argument("--seed", type=int, default=None, help="Random seed")
    solve.add_argument("--mip-gap", type=float, default=None, help="MIP relative gap")
    solve.add_argument("--num-reads", type=int, default=100, help="QUBO sampler reads (neal)")
    solve.add_argument("--sweeps", type=int, default=1000, help="QUBO sampler sweeps (neal)")
    solve.add_argument("--reps", type=int, default=2, help="QAOA depth/repetitions (qaoa)")
    solve.add_argument(
        "--optimizer",
        type=str,
        default="COBYLA",
        help="QAOA optimizer: COBYLA|SPSA|SLSQP (qaoa)",
    )
    solve.add_argument("--num-samples", type=int, default=1000, help="QAOA shots/samples (qaoa)")
    solve.add_argument("--runs-dir", type=Path, default=runs_root(), help="Run record directory")
    solve.set_defaults(func=cmd_solve)

    init = sub.add_parser("init-registry", help="Initialize local registry and seed built-in decision plugins")
    init.add_argument("--force", action="store_true", help="Overwrite existing seeded examples")
    init.set_defaults(func=cmd_init_registry)

    init_agent = sub.add_parser(
        "init-agent",
        help="Scaffold agent-first workspace files (AGENTS.md + decision template)",
    )
    init_agent.add_argument("--path", type=Path, default=Path("."), help="Workspace folder to scaffold")
    init_agent.add_argument("--overwrite", action="store_true", help="Overwrite existing bootstrap files")
    init_agent.set_defaults(func=cmd_init_agent)

    agent_run = sub.add_parser("agent-run", help="Run a structured agent request JSON end-to-end")
    agent_run.add_argument("request", type=Path, help="Path to request JSON")
    agent_run.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for machine-readable result JSON",
    )
    agent_run.set_defaults(func=cmd_agent_run)

    plugin = sub.add_parser("plugin", help="Manage decision plugins locally and on hub")
    plugin_sub = plugin.add_subparsers(dest="plugin_command")

    plugin_list = plugin_sub.add_parser("list", help="List installed decision plugins")
    plugin_list.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    plugin_list.set_defaults(func=cmd_plugin_list)

    plugin_dev_list = plugin_sub.add_parser("dev-list", help="List local workspace decision plugins")
    plugin_dev_list.add_argument(
        "--path",
        type=Path,
        default=Path("decision_plugins"),
        help="Workspace decision plugin root directory",
    )
    plugin_dev_list.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    plugin_dev_list.set_defaults(func=cmd_plugin_dev_list)

    plugin_status = plugin_sub.add_parser("status", help="Compare installed and workspace state for one plugin")
    plugin_status.add_argument("name", type=str, help="Decision plugin name")
    plugin_status.add_argument(
        "--path",
        type=Path,
        default=Path("decision_plugins"),
        help="Workspace decision plugin root directory",
    )
    plugin_status.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    plugin_status.set_defaults(func=cmd_plugin_status)

    plugin_install = plugin_sub.add_parser("install", help="Install a local decision plugin folder")
    plugin_install.add_argument("source", type=Path, help="Path to decision plugin folder")
    plugin_install.add_argument("--overwrite", action="store_true", help="Overwrite if plugin already exists")
    plugin_install.set_defaults(func=cmd_plugin_install)

    plugin_export = plugin_sub.add_parser("export", help="Export an installed decision plugin folder")
    plugin_export.add_argument("name", type=str, help="Installed decision plugin name")
    plugin_export.add_argument("destination", type=Path, help="Destination folder")
    plugin_export.set_defaults(func=cmd_plugin_export)

    plugin_remove = plugin_sub.add_parser("remove", help="Remove an installed decision plugin")
    plugin_remove.add_argument("name", type=str, help="Installed decision plugin name")
    plugin_remove.set_defaults(func=cmd_plugin_remove)

    plugin_push = plugin_sub.add_parser("push", help="Upload a local decision plugin to Rastion Hub")
    plugin_push.add_argument("path", type=Path, help="Path to local decision plugin folder")
    plugin_push.set_defaults(func=cmd_plugin_push)

    plugin_pull = plugin_sub.add_parser("pull", help="Download a decision plugin from Rastion Hub")
    plugin_pull.add_argument("name", type=str, help="Hub decision plugin name")
    plugin_pull.add_argument("--overwrite", action="store_true", help="Overwrite local plugin if it exists")
    plugin_pull.set_defaults(func=cmd_plugin_pull)

    plugin_search = plugin_sub.add_parser("search", help="Search decision plugins on Rastion Hub")
    plugin_search.add_argument("query", nargs="?", default="", help="Search query (blank lists latest matches)")
    plugin_search.set_defaults(func=cmd_plugin_search)

    install_solver = sub.add_parser("install-solver", help="Install a solver plugin from ZIP URL")
    install_solver.add_argument("url", type=str, help="GitHub ZIP URL, e.g. https://github.com/user/repo/archive/main.zip")
    install_solver.add_argument("--name", type=str, default=None, help="Optional local solver folder name")
    install_solver.add_argument("--overwrite", action="store_true", help="Overwrite if solver already exists")
    install_solver.set_defaults(func=cmd_install_solver)

    bench = sub.add_parser("benchmark", help="Benchmark one decision plugin with multiple solvers")
    bench.add_argument("decision_plugin", type=str, help="Decision plugin path or registry name")
    bench.add_argument("--instance", type=str, default="default", help="Instance path or registry instance name")
    bench.add_argument(
        "--solvers",
        nargs="+",
        default=None,
        help="Solver names. Defaults to all available solvers.",
    )
    bench.add_argument("--time-limit", type=str, default="30s", help="Time limit per run, e.g. 30s")
    bench.add_argument("--runs", type=int, default=3, help="Runs per solver")
    bench.set_defaults(func=cmd_benchmark)

    runs = sub.add_parser("runs", help="Inspect local run history")
    runs_sub = runs.add_subparsers(dest="runs_command")

    runs_list = runs_sub.add_parser("list", help="List recent run records")
    runs_list.add_argument("--limit", type=int, default=20, help="Number of recent runs to show")
    runs_list.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    runs_list.set_defaults(func=cmd_runs_list)

    runs_show = runs_sub.add_parser("show", help="Show one run record by 1-based run id")
    runs_show.add_argument("run_id", type=int, help="1-based run id from `rastion runs list`")
    runs_show.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    runs_show.set_defaults(func=cmd_runs_show)

    login = sub.add_parser("login", help="Login to hub via GitHub OAuth browser flow")
    login.set_defaults(func=cmd_login)

    logout = sub.add_parser("logout", help="Remove local hub token")
    logout.set_defaults(func=cmd_logout)

    whoami = sub.add_parser("whoami", help="Show current hub user")
    whoami.set_defaults(func=cmd_whoami)

    return parser


def _format_capabilities(plugin: object) -> str:
    cap = plugin.capabilities()
    vars_supported = ",".join(sorted(v.value for v in cap.variable_types)) or "none"
    obj_supported = ",".join(sorted(o.value for o in cap.objective_types)) or "none"
    cons_supported = ",".join(sorted(c.value for c in cap.constraint_types)) or "none"
    modes = ",".join(sorted(m.value for m in cap.modes)) or "none"
    return f"vars={vars_supported} objective={obj_supported} constraints={cons_supported} modes={modes}"


def _load_spec_instance(spec_ref: str, instance_ref: str) -> tuple[ProblemSpec, InstanceData]:
    spec_candidate = Path(spec_ref)
    if spec_candidate.exists():
        spec = ProblemSpec.from_json_file(spec_candidate)
        instance_candidate = Path(instance_ref)
        instance = InstanceData.from_path(instance_candidate)
        return spec, instance

    plugin = DecisionPlugin.from_registry(spec_ref)
    spec = plugin.spec
    instance = plugin.load_instance(instance_ref)
    return spec, instance


def cmd_info(args: argparse.Namespace) -> int:
    if (args.spec is None) != (args.instance is None):
        print("Provide both spec and instance, or neither.")
        return 1

    init_registry()
    plugins = discover_plugins()
    print(f"rastion {__version__}")
    if not plugins:
        print("No solver plugins installed.")
        return 0

    print("Installed solver plugins:")
    for name in sorted(plugins):
        plugin = plugins[name]
        if args.verbose:
            print(f"- {plugin.name} {plugin.version}")
            print("  " + _format_capabilities(plugin))
        else:
            print(f"- {plugin.name} {plugin.version}")

    if args.verbose and args.spec is not None and args.instance is not None:
        spec, instance = _load_spec_instance(args.spec, args.instance)
        result = validate_problem_and_instance(spec, instance)
        if not result.valid:
            print("\nModel compatibility check skipped: validation failed")
            for error in result.errors:
                print(f"- {error}")
            return 1

        ir_model = compile_to_ir(spec, instance)
        requirements = requirements_from_ir(ir_model)
        print("\nDecision plugin requirements:")
        print("- " + required_capabilities_summary(requirements))

        print("\nCompatibility by solver:")
        for name in sorted(plugins):
            plugin = plugins[name]
            report = match_report(requirements, name, plugin.capabilities())
            if report.matched:
                print(f"- {name}: MATCH (score={report.score})")
            else:
                missing = "; ".join(report.missing)
                print(f"- {name}: NO MATCH (score={report.score})")
                print(f"  missing: {missing}")

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    spec, instance = _load_spec_instance(args.spec, args.instance)
    result = validate_problem_and_instance(spec, instance)

    if result.valid:
        print("Validation: OK")
        for warning in result.warnings:
            print(f"Warning: {warning}")
        return 0

    print("Validation: FAILED")
    for error in result.errors:
        print(f"- {error}")
    return 1


def cmd_solve(args: argparse.Namespace) -> int:
    init_registry()

    spec, instance = _load_spec_instance(args.spec, args.instance)
    validation = validate_problem_and_instance(spec, instance)
    if not validation.valid:
        print("Validation: FAILED")
        for error in validation.errors:
            print(f"- {error}")
        return 1

    ir_model = compile_to_ir(spec, instance)
    plugins = discover_plugins()
    if not plugins:
        print("No solver plugins available.")
        return 1

    solver_config: dict[str, object] = {
        "time_limit": args.time_limit,
        "seed": args.seed,
        "mip_gap": args.mip_gap,
        "num_reads": args.num_reads,
        "sweeps": args.sweeps,
        "reps": args.reps,
        "optimizer": args.optimizer,
        "num_samples": args.num_samples,
    }
    solver_config = {k: v for k, v in solver_config.items() if v is not None}

    backend = LocalBackend()

    if args.solver == "auto":
        preferred = list(args.preferred)
        candidates = compatible_plugins(ir_model, plugins, preferred=preferred)
        if not candidates:
            try:
                auto_select_solver(ir_model, plugins, preferred=preferred)
            except ValueError as exc:
                print(str(exc))
            return 1

        selected_plugin = None
        last_error: Exception | None = None
        solution: Solution | None = None
        warned: set[str] = set()
        for plugin in candidates:
            selected_plugin = plugin
            if plugin.name not in warned:
                _warn_if_hub_solver(plugin.name)
                warned.add(plugin.name)
            try:
                solution = backend.run(plugin, ir_model, solver_config)
                if solution.status == SolutionStatus.ERROR:
                    last_error = RuntimeError(solution.error_message or "solver returned ERROR status")
                    solution = None
                    continue
                break
            except Exception as exc:  # pragma: no cover - depends on optional solvers
                last_error = exc
                solution = None
                continue

        if solution is None:
            assert selected_plugin is not None
            print("All compatible solvers failed in auto mode.")
            if last_error is not None:
                print(f"Last error: {last_error}")
            return 1
        solver_plugin = selected_plugin
    else:
        if args.solver not in plugins:
            print(f"Requested solver '{args.solver}' is not installed.")
            print("Installed solvers: " + ", ".join(sorted(plugins)))
            return 1
        solver_plugin = plugins[args.solver]
        _warn_if_hub_solver(solver_plugin.name)
        try:
            solution = backend.run(solver_plugin, ir_model, solver_config)
        except Exception as exc:
            solution = Solution(
                status=SolutionStatus.ERROR,
                objective_value=None,
                primal_values={},
                metadata={
                    "runtime_s": 0.0,
                    "solver_name": solver_plugin.name,
                    "solver_version": solver_plugin.version,
                },
                error_message=str(exc),
            )

    assert solution is not None

    run_record = create_run_record(
        spec=spec,
        instance=instance,
        solution=solution,
        solver_name=solver_plugin.name,
        solver_version=solver_plugin.version,
        solver_config=solver_config,
    )
    out_file = append_run_record(run_record, runs_dir=args.runs_dir)

    print(f"Solver: {solver_plugin.name} {solver_plugin.version}")
    print(f"Status: {solution.status.value}")
    if solution.objective_value is not None:
        print(f"Objective: {solution.objective_value:.6g}")
    if solution.error_message:
        print(f"Error: {solution.error_message}")
    runtime = solution.metadata.get("runtime_s")
    if runtime is not None:
        print(f"Runtime: {runtime:.6g}s")

    if solution.primal_values:
        preview = dict(list(solution.primal_values.items())[:10])
        print("Primal sample:")
        print(json.dumps(preview, indent=2, sort_keys=True))

    print(f"RunRecord: {out_file}")

    return 0 if solution.status in {SolutionStatus.OPTIMAL, SolutionStatus.FEASIBLE} else 1


def cmd_init_registry(args: argparse.Namespace) -> int:
    root = init_registry(force=args.force)
    count = len(list_decision_plugins())
    print(f"Initialized local registry at: {root}")
    print(f"Decision plugins available: {count}")
    return 0


def cmd_init_agent(args: argparse.Namespace) -> int:
    try:
        created = init_agent_workspace(args.path, overwrite=args.overwrite)
    except Exception as exc:
        print(f"Agent bootstrap failed: {exc}")
        return 1

    root = args.path.expanduser().resolve()
    print(f"Initialized agent workspace at: {root}")
    print("Created files:")
    for path in created:
        print(f"- {path}")
    print("Next steps:")
    print("- Review AGENTS.md and adapt policy to your team.")
    print("- Convert real decision data into decision_plugins/<name>/instances/default.json.")
    print("- Run `rastion agent-run ./agent_requests/calendar_week_off_30m.json` for full orchestration.")
    print(
        "- Run `rastion plugin install ./decision_plugins/<name> --overwrite` "
        "then `rastion solve <name> default --solver auto`."
    )
    return 0


def cmd_agent_run(args: argparse.Namespace) -> int:
    try:
        result = run_agent_request_file(args.request, output_json=args.output_json)
    except Exception as exc:
        print(f"Agent run failed: {exc}")
        return 1

    plugin_name = str(result["decision_plugin"])
    print(f"Decision Plugin: {plugin_name} / {result['instance']}")
    solver = result.get("solver", {})
    print(f"Solver: {solver.get('name')} {solver.get('version')}")
    print(f"Status: {result['status']}")
    if result.get("objective_value") is not None:
        print(f"Objective: {result['objective_value']:.6g}")
    if result.get("runtime_s") is not None:
        print(f"Runtime: {result['runtime_s']:.6g}s")
    installed = result.get("installed_solver_folders") or []
    if installed:
        print("Installed solvers:")
        for name in installed:
            print(f"- {name}")
    if result.get("error_message"):
        print(f"Error: {result['error_message']}")
    if result.get("output_json"):
        print(f"Output JSON: {result['output_json']}")
    print(f"RunRecord: {result['run_record_file']}")

    return 0 if result["status"] in {SolutionStatus.OPTIMAL.value, SolutionStatus.FEASIBLE.value} else 1


def cmd_plugin_list(args: argparse.Namespace) -> int:
    init_registry(copy_examples=False)
    plugins = _installed_decision_plugin_entries()
    if args.json:
        print(json.dumps(plugins, indent=2, sort_keys=True))
        return 0

    if not plugins:
        print("No decision plugins installed.")
        return 0

    print("Installed decision plugins:")
    for plugin in plugins:
        print(f"- {plugin['name']} {plugin['version']} ({plugin['path']})")
    return 0


def cmd_plugin_dev_list(args: argparse.Namespace) -> int:
    plugins = _workspace_decision_plugin_entries(args.path)
    if args.json:
        print(json.dumps(plugins, indent=2, sort_keys=True))
        return 0

    if not plugins:
        print(f"No local decision plugins found under {args.path.expanduser().resolve()}.")
        return 0

    print("Local workspace decision plugins:")
    for plugin in plugins:
        print(f"- {plugin['name']} {plugin['version']} ({plugin['path']})")
    return 0


def cmd_plugin_status(args: argparse.Namespace) -> int:
    installed = [entry for entry in _installed_decision_plugin_entries() if entry["name"] == args.name]
    workspace = [
        entry
        for entry in _workspace_decision_plugin_entries(args.path)
        if entry["name"] == args.name or Path(entry["path"]).name == args.name
    ]

    payload: dict[str, Any] = {
        "name": args.name,
        "installed": installed[0] if installed else None,
        "workspace": workspace[0] if workspace else None,
        "in_sync": None,
    }
    if payload["installed"] and payload["workspace"]:
        payload["in_sync"] = payload["installed"]["fingerprint"] == payload["workspace"]["fingerprint"]

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"Decision plugin: {args.name}")
    if payload["installed"] is None:
        print("- Installed: no")
    else:
        installed_entry = payload["installed"]
        print(f"- Installed: yes ({installed_entry['version']}, {installed_entry['path']})")
    if payload["workspace"] is None:
        print("- Workspace: no")
    else:
        workspace_entry = payload["workspace"]
        print(f"- Workspace: yes ({workspace_entry['version']}, {workspace_entry['path']})")
    if payload["in_sync"] is not None:
        print(f"- In Sync: {'yes' if payload['in_sync'] else 'no'}")
    return 0


def cmd_plugin_install(args: argparse.Namespace) -> int:
    try:
        target = install_decision_plugin(args.source, overwrite=args.overwrite)
    except Exception as exc:
        print(f"Decision plugin install failed: {exc}")
        return 1

    print(f"Installed decision plugin at {target}")
    return 0


def cmd_plugin_export(args: argparse.Namespace) -> int:
    try:
        target = export_decision_plugin(args.name, args.destination)
    except Exception as exc:
        print(f"Decision plugin export failed: {exc}")
        return 1

    print(f"Exported decision plugin '{args.name}' to {target}")
    return 0


def cmd_plugin_remove(args: argparse.Namespace) -> int:
    removed = remove_decision_plugin(args.name)
    if not removed:
        print(f"Decision plugin '{args.name}' not found.")
        return 1
    print(f"Removed decision plugin '{args.name}'.")
    return 0


def cmd_plugin_push(args: argparse.Namespace) -> int:
    try:
        client = HubClient()
        result = asyncio.run(client.push_decision_plugin(args.path))
    except (HubClientError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Decision plugin push failed: {exc}")
        if _is_hub_auth_error(exc):
            print("Run `rastion login` and retry.")
        return 1

    print(f"Pushed decision plugin '{result.get('name', args.path.name)}' to {client.base_url}")
    return 0


def cmd_plugin_pull(args: argparse.Namespace) -> int:
    try:
        client = HubClient()
        result = asyncio.run(client.pull_decision_plugin(args.name, overwrite=args.overwrite))
    except (HubClientError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Decision plugin pull failed: {exc}")
        return 1

    print(f"Pulled decision plugin '{result.get('name', args.name)}' into {result.get('path', '')}")
    return 0


def cmd_plugin_search(args: argparse.Namespace) -> int:
    try:
        client = HubClient()
        plugins = asyncio.run(client.list_decision_plugins(args.query))
    except (HubClientError, RuntimeError, ValueError) as exc:
        print(f"Decision plugin search failed: {exc}")
        return 1

    if not plugins:
        print("No decision plugins found on hub.")
        return 0

    print("Hub decision plugins:")
    for item in plugins:
        name = str(item.get("name", "unknown"))
        version = str(item.get("version", "n/a"))
        owner = _hub_owner(item)
        downloads = item.get("download_count", "n/a")
        rating = item.get("rating", "n/a")
        print(f"- {name} v{version} ({owner}) downloads={downloads} rating={rating}")
    return 0


def cmd_runs_list(args: argparse.Namespace) -> int:
    indexed = _indexed_run_records()
    if args.limit > 0:
        indexed = indexed[-args.limit :]

    rows = [_run_summary(index, record) for index, record in indexed]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0

    if not rows:
        print("No run history found.")
        return 0

    print("Recent runs:")
    for row in rows:
        objective = "n/a" if row["objective_value"] is None else f"{row['objective_value']}"
        print(
            f"- #{row['run_id']} {row['timestamp']} solver={row['solver']} "
            f"status={row['status']} objective={objective}"
        )
    return 0


def cmd_runs_show(args: argparse.Namespace) -> int:
    indexed = _indexed_run_records()
    if args.run_id < 1 or args.run_id > len(indexed):
        print(f"Run id out of range: {args.run_id}. Available range: 1..{len(indexed)}")
        return 1

    _, record = indexed[args.run_id - 1]
    if args.json:
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0

    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def cmd_install_solver(args: argparse.Namespace) -> int:
    try:
        target = install_solver_from_url(args.url, name=args.name, overwrite=args.overwrite)
    except Exception as exc:
        print(f"Install solver failed: {exc}")
        return 1

    print(f"Installed solver at {target}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    try:
        results = compare(
            args.decision_plugin,
            instance=args.instance,
            solvers=args.solvers,
            time_limit=args.time_limit,
            runs=args.runs,
        )
    except Exception as exc:
        print(f"Decision plugin benchmark failed: {exc}")
        return 1

    for result in results:
        objective = "n/a" if result.objective is None else f"{result.objective:.6g}"
        print(
            f"{result.solver}: status={result.status} objective={objective} "
            f"runtime={result.runtime:.6g}s gap={result.gap} success={result.succeeded_runs}/{result.runs}"
        )
        for err in result.errors:
            print(f"  error: {err}")

    return 0


def cmd_login(args: argparse.Namespace) -> int:
    _ = args
    client = HubClient()

    try:
        oauth_url = asyncio.run(client.oauth_login_url())
    except (HubClientError, RuntimeError, ValueError) as exc:
        if isinstance(exc, HubClientError) and _oauth_endpoint_unsupported(exc):
            print("Hub does not support browser OAuth login; falling back to token login.")
            return _complete_login_with_token(client, prompt="GitHub token: ")
        print(f"Login failed: {exc}")
        return 1

    print("Opening browser for GitHub authorization...")
    opened, opener = _open_url_in_browser(oauth_url)
    if opened:
        if opener:
            print(f"Browser launch command: {opener}")
    else:
        print("Unable to open browser automatically in this terminal session.")
    print(f"Authorize here: {oauth_url}")

    return _complete_login_with_token(client, prompt="Paste token from callback page: ")


def cmd_logout(args: argparse.Namespace) -> int:
    _ = args
    try:
        client = HubClient()
        asyncio.run(client.logout())
    except (HubClientError, RuntimeError, ValueError) as exc:
        print(f"Logout failed: {exc}")
        return 1
    print("Logged out.")
    return 0


def cmd_whoami(args: argparse.Namespace) -> int:
    _ = args
    token = get_token()
    if not token:
        print("Not logged in. Run `rastion login`.")
        return 1

    try:
        client = HubClient(token=token)
        user = asyncio.run(client.get_user())
    except (HubClientError, RuntimeError, ValueError) as exc:
        print(f"whoami failed: {exc}")
        return 1

    username = str(user.get("username") or user.get("login") or "unknown")
    user_id = user.get("id")
    if user_id is not None:
        print(f"{username} (id={user_id})")
    else:
        print(username)
    return 0


def _warn_if_hub_solver(name: str) -> None:
    source = solver_hub_source(name)
    if source is None:
        return

    print(f'WARNING: You are about to run solver "{name}" from the hub.')
    print("This is Python code from an untrusted source. Only run if you trust the author.")


def _is_hub_auth_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    return "not authenticated" in message or "unauthorized" in message or "401" in message


def _hub_owner(item: dict[str, object]) -> str:
    owner = item.get("owner")
    if isinstance(owner, dict):
        username = owner.get("username")
        if isinstance(username, str) and username.strip():
            return username
    return "unknown"


def _installed_decision_plugin_entries() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for entry in list_decision_plugins():
        plugin_path = Path(entry.path) if entry.path is not None else (problems_root() / entry.name)
        entries.append(
            {
                "name": entry.name,
                "version": entry.version,
                "author": entry.author,
                "path": str(plugin_path.resolve()),
                "fingerprint": _decision_plugin_fingerprint(plugin_path),
            }
        )
    entries.sort(key=lambda item: (str(item["name"]), str(item["version"]), str(item["path"])))
    return entries


def _workspace_decision_plugin_entries(path: Path) -> list[dict[str, object]]:
    root = path.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return []

    entries: list[dict[str, object]] = []
    for candidate in sorted(root.iterdir(), key=lambda p: p.name):
        if not candidate.is_dir():
            continue
        spec_path = candidate / "spec.json"
        if not spec_path.exists():
            continue

        name = candidate.name
        version = "0.1.0"
        author = "unknown"
        metadata_path = candidate / "metadata.yaml"
        if metadata_path.exists():
            loaded = read_yaml_file(metadata_path)
            if isinstance(loaded, dict):
                raw_name = loaded.get("name")
                raw_version = loaded.get("version")
                raw_author = loaded.get("author")
                if isinstance(raw_name, str) and raw_name.strip():
                    name = raw_name.strip()
                if isinstance(raw_version, (str, int, float)):
                    version = str(raw_version).strip() or version
                if isinstance(raw_author, str) and raw_author.strip():
                    author = raw_author.strip()

        entries.append(
            {
                "name": name,
                "version": version,
                "author": author,
                "path": str(candidate),
                "fingerprint": _decision_plugin_fingerprint(candidate),
            }
        )

    entries.sort(key=lambda item: (str(item["name"]), str(item["version"]), str(item["path"])))
    return entries


def _decision_plugin_fingerprint(root: Path) -> str:
    target = root.expanduser().resolve()
    digest = hashlib.sha256()
    files: list[Path] = []
    for relative in ("spec.json", "metadata.yaml", "problem_card.md", "decision.yaml"):
        candidate = target / relative
        if candidate.exists() and candidate.is_file():
            files.append(candidate)
    instances_dir = target / "instances"
    if instances_dir.exists() and instances_dir.is_dir():
        for candidate in sorted(instances_dir.iterdir(), key=lambda p: p.name):
            if candidate.is_file() and candidate.suffix.lower() in {".json", ".npz"}:
                files.append(candidate)

    for candidate in sorted(files, key=lambda p: str(p.relative_to(target))):
        rel = str(candidate.relative_to(target)).replace("\\", "/")
        digest.update(rel.encode("utf-8"))
        digest.update(b"\n")
        digest.update(candidate.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _indexed_run_records() -> list[tuple[int, dict[str, Any]]]:
    history = runs_file()
    if not history.exists() or not history.is_file():
        return []

    records: list[tuple[int, dict[str, Any]]] = []
    for line in history.read_text(encoding="utf-8").splitlines():
        payload = line.strip()
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        records.append((len(records) + 1, parsed))
    return records


def _run_summary(run_id: int, record: dict[str, Any]) -> dict[str, Any]:
    solution = record.get("solution")
    if not isinstance(solution, dict):
        solution = {}
    return {
        "run_id": run_id,
        "timestamp": record.get("timestamp"),
        "solver": record.get("solver_name"),
        "status": solution.get("status"),
        "objective_value": solution.get("objective_value"),
    }


def _complete_login_with_token(client: HubClient, *, prompt: str) -> int:
    try:
        entered = input(prompt).strip()
    except EOFError:
        print("Login failed: no token provided.")
        return 1

    token = entered
    if not token:
        token = _env_github_token() or ""
        if token:
            print("Using GitHub token from local .env/environment.")

    if not token:
        print("Login failed: token cannot be empty and no token was found in local .env/environment.")
        return 1

    try:
        user = asyncio.run(client.login(token))
    except (HubClientError, RuntimeError, ValueError) as exc:
        print(f"Login failed: {exc}")
        return 1

    username = str(user.get("username") or user.get("login") or "unknown")
    print(f"Logged in as {username}")
    return 0


def _env_github_token() -> str | None:
    for key in ("RASTION_GITHUB_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for dotenv_path in _dotenv_candidates():
        token = _read_dotenv_token(dotenv_path)
        if token:
            return token
    return None


def _dotenv_candidates() -> list[Path]:
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    ordered: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _read_dotenv_token(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            normalized_key = key.strip()
            if normalized_key not in {"RASTION_GITHUB_TOKEN", "GITHUB_TOKEN"}:
                continue
            cleaned = value.strip().strip('"').strip("'")
            if cleaned:
                return cleaned
    except OSError:
        return None
    return None


def _oauth_endpoint_unsupported(exc: HubClientError) -> bool:
    message = str(exc).lower()
    if "method not allowed" in message or "not found" in message:
        return True
    return re.search(r"\b(404|405)\b", message) is not None


def _open_url_in_browser(url: str) -> tuple[bool, str | None]:
    try:
        if webbrowser.open(url, new=2):
            return True, "python-webbrowser"
    except Exception:
        pass

    candidates: list[tuple[str, list[str]]] = []
    if shutil.which("wslview"):
        candidates.append(("wslview", ["wslview", url]))
    if shutil.which("xdg-open"):
        candidates.append(("xdg-open", ["xdg-open", url]))
    if shutil.which("gio"):
        candidates.append(("gio open", ["gio", "open", url]))
    if shutil.which("open"):
        candidates.append(("open", ["open", url]))
    if os.environ.get("WSL_DISTRO_NAME") and shutil.which("cmd.exe"):
        candidates.append(("cmd.exe start", ["cmd.exe", "/C", "start", "", url]))

    for label, command in candidates:
        try:
            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True, label
        except OSError:
            continue

    return False, None


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    init_registry()

    parser = build_parser()
    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
