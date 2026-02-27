"""Interactive benchmark flow."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from rastion.benchmark import compare
from rastion.registry.loader import Solver
from rastion.registry.manager import list_problems
from rastion.tui.controls import prompt_input, select_multiple, select_option, wait_for_key


def benchmark_solvers(console: Console) -> None:
    console.clear()
    problems = list_problems()
    if not problems:
        console.print("[yellow]No registered problems found.[/yellow]")
        wait_for_key(console, title="Benchmark", prompt="No registered problems found.")
        return

    problem_names = [entry.name for entry in problems]
    problem_idx = select_option(
        console,
        title="Benchmark",
        prompt="Choose a problem:",
        options=problem_names,
    )
    if problem_idx is None:
        return
    problem_name = problem_names[problem_idx]

    available = Solver.available()
    selected_indices = select_multiple(
        console,
        title="Benchmark",
        prompt="Select one or more solvers:",
        options=available,
    )
    if selected_indices is None:
        return
    if not selected_indices:
        console.clear()
        console.print("[yellow]No solvers selected.[/yellow]")
        wait_for_key(console, title="Benchmark", prompt="Benchmark canceled.")
        return

    selected_solvers = [available[idx] for idx in selected_indices]

    runs_text = prompt_input(
        console,
        title="Benchmark",
        prompt="Number of runs per solver",
        default="3",
    )
    if runs_text is None:
        return

    limit_text = prompt_input(
        console,
        title="Benchmark",
        prompt="Time limit (e.g. 30s)",
        default="30s",
    )
    if limit_text is None:
        return

    try:
        runs = max(1, int(runs_text or "1"))
    except ValueError:
        runs = 1

    results = compare(
        problem_name,
        instance="default",
        solvers=list(selected_solvers),
        time_limit=limit_text or "30s",
        runs=runs,
    )

    table = Table(title=f"Benchmark: {problem_name}")
    table.add_column("Solver", style="cyan")
    table.add_column("Status")
    table.add_column("Objective")
    table.add_column("Runtime (avg s)")
    table.add_column("Gap")
    table.add_column("Success")

    for result in results:
        objective = "" if result.objective is None else f"{result.objective:.6g}"
        gap = "" if result.gap is None else f"{result.gap:.6g}"
        table.add_row(
            result.solver,
            result.status,
            objective,
            f"{result.runtime:.6g}",
            gap,
            f"{result.succeeded_runs}/{result.runs}",
        )

    console.clear()
    console.print(table)
    wait_for_key(console, title="Benchmark", prompt="Benchmark run completed.")
