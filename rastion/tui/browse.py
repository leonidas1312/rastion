"""Browse problem and solver registry views."""

from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from rastion.hub import HubClient, HubClientError
from rastion.registry.loader import Problem, Solver
from rastion.registry.manager import list_problems, list_solvers
from rastion.tui.controls import prompt_input, select_option, wait_for_key


def browse_problems(console: Console) -> None:
    console.clear()
    entries = list_problems()
    if not entries:
        console.print("[yellow]No registered problems found.[/yellow]")
        wait_for_key(console, title="Browse Problems", prompt="No registered problems found.")
        return

    table = Table(title="Registered Problems", show_lines=False)
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Class")
    table.add_column("Difficulty")
    table.add_column("Tags")
    for entry in entries:
        table.add_row(
            entry.name,
            entry.version,
            entry.optimization_class,
            entry.difficulty,
            ", ".join(entry.tags),
        )

    option_names = [entry.name for entry in entries]
    choice_idx = select_option(
        console,
        title="Browse Problems",
        prompt="Select a problem to view details:",
        options=option_names,
        extra_renderables=[table],
    )
    if choice_idx is None:
        return
    choice = option_names[choice_idx]

    problem = Problem.from_registry(choice)
    metadata = problem.metadata

    console.clear()
    meta_table = Table(title=f"Metadata: {choice}")
    meta_table.add_column("Field", style="green")
    meta_table.add_column("Value", style="white")
    for key in ("name", "version", "author", "optimization_class", "difficulty", "tags"):
        value = metadata.get(key)
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        meta_table.add_row(key, str(value))

    instances = ", ".join(problem.instances) or "none"
    meta_table.add_row("instances", instances)
    console.print(meta_table)
    console.print(Panel(Markdown(problem.card), title="Problem Card", border_style="blue"))

    wait_for_key(console, title="Browse Problems", prompt="Problem details shown.")


def browse_solvers(console: Console) -> None:
    source_idx = select_option(
        console,
        title="Browse Solvers",
        prompt="Choose source:",
        options=["Local solvers", "Hub solvers"],
    )
    if source_idx is None:
        return

    if source_idx == 0:
        _browse_local_solvers(console)
    else:
        _browse_hub_solvers(console)


def _browse_local_solvers(console: Console) -> None:
    console.clear()
    entries = list_solvers()
    if not entries:
        console.print("[yellow]No solver plugins available.[/yellow]")
        wait_for_key(console, title="Browse Solvers", prompt="No solver plugins available.")
        return

    table = Table(title="Installed Solvers")
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Open Source")
    table.add_column("Modes")
    table.add_column("Objectives")

    for entry in entries:
        cap = entry.capabilities
        modes = ", ".join(cap.get("modes", []))
        objectives = ", ".join(cap.get("objective_types", []))
        table.add_row(entry.name, entry.version, str(entry.open_source), modes, objectives)

    option_names = [entry.name for entry in entries]
    choice_idx = select_option(
        console,
        title="Browse Solvers",
        prompt="Select a solver for capability details:",
        options=option_names,
        extra_renderables=[table],
    )
    if choice_idx is None:
        return
    choice = option_names[choice_idx]

    solver = Solver.from_name(choice)
    console.clear()
    console.print(Panel(Markdown(solver.card), title="Solver Card", border_style="magenta"))
    wait_for_key(console, title="Browse Solvers", prompt="Solver details shown.")


def _browse_hub_solvers(console: Console) -> None:
    query = prompt_input(
        console,
        title="Browse Hub Solvers",
        prompt="Search query (blank shows all)",
        default="",
    )
    if query is None:
        return

    try:
        client = HubClient()
        results = asyncio.run(client.search(query.strip(), type="solver"))
    except (HubClientError, OSError, RuntimeError, ValueError) as exc:
        console.clear()
        console.print(f"[red]Hub search failed: {exc}[/red]")
        wait_for_key(console, title="Browse Hub Solvers", prompt="Hub search failed.")
        return

    solvers = results.get("solvers", [])
    if not solvers:
        console.clear()
        console.print("[yellow]No solver matches found on hub.[/yellow]")
        wait_for_key(console, title="Browse Hub Solvers", prompt="No solver matches found.")
        return

    table = Table(title="Hub Solvers")
    table.add_column("Name", style="magenta")
    table.add_column("Version")
    table.add_column("Owner")
    table.add_column("Downloads")
    table.add_column("Rating")

    for item in solvers:
        table.add_row(
            str(item.get("name", "")),
            str(item.get("version", "")),
            _hub_owner(item),
            str(item.get("download_count", "n/a")),
            str(item.get("rating", "n/a")),
        )

    console.clear()
    console.print(table)
    wait_for_key(console, title="Browse Hub Solvers", prompt="Hub solver list loaded.")


def _hub_owner(item: dict[str, Any]) -> str:
    owner = item.get("owner")
    if isinstance(owner, dict):
        username = owner.get("username")
        if isinstance(username, str) and username.strip():
            return username
    return "unknown"
