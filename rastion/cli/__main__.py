"""Command line interface for rastion."""

from __future__ import annotations

import asyncio
import argparse
import json
import sys
import webbrowser
from pathlib import Path

from rastion.backends.local import LocalBackend
from rastion.benchmark import compare
from rastion.config import get_token
from rastion.compile.normalize import compile_to_ir
from rastion.core.data import InstanceData
from rastion.core.run_record import append_run_record, create_run_record
from rastion.core.solution import Solution, SolutionStatus
from rastion.core.spec import ProblemSpec
from rastion.core.validate import validate_problem_and_instance
from rastion.registry.loader import Problem
from rastion.registry.manager import (
    export_problem,
    init_registry,
    install_problem,
    install_solver_from_url,
    list_problems,
    runs_root,
    solver_hub_source,
)
from rastion.hub import HubClient, HubClientError
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
        description="Portable optimization runner.",
    )
    parser.add_argument("--version", action="version", version=f"rastion {__version__}")

    sub = parser.add_subparsers(dest="command")

    info = sub.add_parser("info", help="Show installed solver plugins and capabilities")
    info.add_argument("--verbose", action="store_true", help="Show capabilities and compatibility details")
    info.add_argument("spec", nargs="?", type=str, help="Problem spec path OR registry problem name")
    info.add_argument("instance", nargs="?", type=str, help="Instance path OR registry instance name")
    info.set_defaults(func=cmd_info)

    validate = sub.add_parser("validate", help="Validate problem spec + instance data")
    validate.add_argument("spec", type=str, help="Path to spec.json OR registry problem name")
    validate.add_argument("instance", type=str, help="Path to instance JSON/NPZ OR registry instance name")
    validate.set_defaults(func=cmd_validate)

    solve = sub.add_parser("solve", help="Solve a problem instance")
    solve.add_argument("spec", type=str, help="Path to spec.json OR registry problem name")
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

    init = sub.add_parser("init-registry", help="Initialize local registry and seed built-in problems")
    init.add_argument("--force", action="store_true", help="Overwrite existing seeded examples")
    init.set_defaults(func=cmd_init_registry)

    export = sub.add_parser("export", help="Export a registry problem folder for sharing")
    export.add_argument("problem", type=str, help="Problem name in local registry")
    export.add_argument("destination", type=Path, help="Destination folder")
    export.set_defaults(func=cmd_export)

    install = sub.add_parser("install", help="Install a local problem folder into registry")
    install.add_argument("source", type=Path, help="Path to shareable problem folder")
    install.add_argument("--overwrite", action="store_true", help="Overwrite if problem already exists")
    install.set_defaults(func=cmd_install)

    install_solver = sub.add_parser("install-solver", help="Install a solver plugin from ZIP URL")
    install_solver.add_argument("url", type=str, help="GitHub ZIP URL, e.g. https://github.com/user/repo/archive/main.zip")
    install_solver.add_argument("--name", type=str, default=None, help="Optional local solver folder name")
    install_solver.add_argument("--overwrite", action="store_true", help="Overwrite if solver already exists")
    install_solver.set_defaults(func=cmd_install_solver)

    bench = sub.add_parser("benchmark", help="Benchmark one problem with multiple solvers")
    bench.add_argument("problem", type=str, help="Problem path or registry name")
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

    problem = Problem.from_registry(spec_ref)
    spec = problem.spec
    instance = problem.load_instance(instance_ref)
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
        print("\nProblem requirements:")
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
    count = len(list_problems())
    print(f"Initialized local registry at: {root}")
    print(f"Problems available: {count}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    try:
        target = export_problem(args.problem, args.destination)
    except Exception as exc:
        print(f"Export failed: {exc}")
        return 1

    print(f"Exported '{args.problem}' to {target}")
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    try:
        target = install_problem(args.source, overwrite=args.overwrite)
    except Exception as exc:
        print(f"Install failed: {exc}")
        return 1

    print(f"Installed problem at {target}")
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
            args.problem,
            instance=args.instance,
            solvers=args.solvers,
            time_limit=args.time_limit,
            runs=args.runs,
        )
    except Exception as exc:
        print(f"Benchmark failed: {exc}")
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
        print(f"Login failed: {exc}")
        return 1

    print("Opening browser for GitHub authorization...")
    opened = webbrowser.open(oauth_url)
    if not opened:
        print("Unable to open browser automatically.")
    print(f"Authorize here: {oauth_url}")

    try:
        token = input("Paste token from callback page: ").strip()
    except EOFError:
        print("Login failed: no token provided.")
        return 1

    if not token:
        print("Login failed: token cannot be empty.")
        return 1

    try:
        user = asyncio.run(client.login(token))
    except (HubClientError, RuntimeError, ValueError) as exc:
        print(f"Login failed: {exc}")
        return 1

    username = str(user.get("username") or user.get("login") or "unknown")
    print(f"Logged in as {username}")
    return 0


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
