"""Interactive solve flow."""

from __future__ import annotations

import json

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from rastion.backends.local import LocalBackend
from rastion.compile.normalize import compile_to_ir
from rastion.core.run_record import append_run_record, create_run_record
from rastion.core.solution import Solution, SolutionStatus
from rastion.registry.loader import Problem, Solver
from rastion.registry.manager import list_problems, runs_root
from rastion.solvers.discovery import discover_plugins
from rastion.solvers.matching import auto_select_solver
from rastion.tui.controls import prompt_input, select_option, wait_for_key


def solve_problem(console: Console) -> None:
    console.clear()
    entries = list_problems()
    if not entries:
        console.print("[yellow]No registered problems found.[/yellow]")
        wait_for_key(console, title="Solve", prompt="No registered problems found.")
        return

    problem_names = [entry.name for entry in entries]
    problem_idx = select_option(
        console,
        title="Solve",
        prompt="Choose a problem:",
        options=problem_names,
    )
    if problem_idx is None:
        return
    problem_name = problem_names[problem_idx]

    problem = Problem.from_registry(problem_name)
    instance_names = problem.instances or ["default"]

    instance_idx = select_option(
        console,
        title="Solve",
        prompt="Choose an instance:",
        options=instance_names,
    )
    if instance_idx is None:
        return
    instance_name = instance_names[instance_idx]

    available_solvers = ["auto", *Solver.available()]
    solver_idx = select_option(
        console,
        title="Solve",
        prompt="Choose solver:",
        options=available_solvers,
    )
    if solver_idx is None:
        return
    solver_name = available_solvers[solver_idx]

    time_limit_text = prompt_input(
        console,
        title="Solve",
        prompt="Time limit in seconds (blank for none)",
        default="30",
    )
    if time_limit_text is None:
        return

    config: dict[str, object] = {}
    if time_limit_text and time_limit_text.strip():
        try:
            config["time_limit"] = float(time_limit_text.strip())
        except ValueError:
            console.clear()
            console.print(f"[yellow]Ignoring invalid time limit: {time_limit_text}[/yellow]")
            wait_for_key(console, title="Solve", prompt="Invalid time limit ignored.")

    spec = problem.spec
    instance = problem.load_instance(instance_name)
    ir_model = compile_to_ir(spec, instance)
    plugins = discover_plugins()

    try:
        selected_plugin = _select_solver_plugin(solver_name, ir_model, plugins)
    except ValueError as exc:
        console.clear()
        console.print(f"[red]{exc}[/red]")
        wait_for_key(console, title="Solve", prompt="Unable to select solver.")
        return

    backend = LocalBackend()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task(description=f"Running {selected_plugin.name}...", total=None)
        solution: Solution
        try:
            solution = backend.run(selected_plugin, ir_model, config)
        except Exception as exc:  # pragma: no cover - optional runtime paths
            solution = Solution(
                status=SolutionStatus.ERROR,
                objective_value=None,
                primal_values={},
                metadata={
                    "runtime_s": 0.0,
                    "solver_name": selected_plugin.name,
                    "solver_version": selected_plugin.version,
                },
                error_message=str(exc),
            )

    run_record = create_run_record(
        spec=spec,
        instance=instance,
        solution=solution,
        solver_name=selected_plugin.name,
        solver_version=selected_plugin.version,
        solver_config=config,
    )
    output_file = append_run_record(run_record, runs_dir=runs_root())

    result_table = Table(title="Solve Result")
    result_table.add_column("Field", style="green")
    result_table.add_column("Value", style="white")
    result_table.add_row("Problem", problem_name)
    result_table.add_row("Instance", instance_name)
    result_table.add_row("Solver", f"{selected_plugin.name} {selected_plugin.version}")
    result_table.add_row("Status", solution.status.value)
    result_table.add_row("Objective", "" if solution.objective_value is None else f"{solution.objective_value:.6g}")
    result_table.add_row("Runtime", str(solution.metadata.get("runtime_s", "n/a")))
    result_table.add_row("RunRecord", str(output_file))

    if solution.error_message:
        result_table.add_row("Error", solution.error_message)

    console.clear()
    console.print(result_table)
    if solution.primal_values:
        preview = dict(list(solution.primal_values.items())[:10])
        console.print("Primal sample:")
        console.print(json.dumps(preview, indent=2, sort_keys=True))

    wait_for_key(console, title="Solve", prompt="Solve run completed.")


def _select_solver_plugin(solver_name: str, ir_model: object, plugins: dict[str, object]) -> object:
    if not plugins:
        raise ValueError("No solver plugins available")

    if solver_name == "auto":
        return auto_select_solver(ir_model, plugins)

    selected = plugins.get(solver_name)
    if selected is None:
        raise ValueError(f"solver '{solver_name}' is not installed")
    return selected
