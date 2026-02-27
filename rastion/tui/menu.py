"""Main TUI menu for Rastion optimization lab."""

from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rastion.hub import HubClient, HubClientError
from rastion.registry.manager import (
    config_file,
    hub_config,
    init_registry,
    install_solver_from_url,
    load_config,
    rastion_home,
    runs_file,
    write_config,
)
from rastion.tui.benchmark import benchmark_solvers
from rastion.tui.browse import browse_problems, browse_solvers
from rastion.tui.controls import prompt_input, select_option, wait_for_key
from rastion.tui.solve import solve_problem

_MENU_CHOICES = [
    ("explore", "1. Explore"),
    ("solve", "2. Solve"),
    ("install", "3. Install"),
    ("share", "4. Share"),
    ("onboarding", "5. Onboarding"),
    ("settings", "6. Settings"),
]

_ONBOARDING_PHRASES = [
    "Better optimization helps hospitals reduce wait times and serve patients faster.",
    "Smarter routing and scheduling cuts fuel waste, cost, and commuting time.",
    "Improved resource allocation means more reliable services with fewer shortages.",
]


def launch_menu() -> int:
    init_registry()
    console = Console(force_terminal=True)
    menu_labels = [label for _, label in _MENU_CHOICES]

    while True:
        choice_idx = select_option(
            console,
            title="Rastion v0.1",
            prompt="Choose an action:",
            options=menu_labels,
            extra_renderables=[_banner_panel()],
        )
        if choice_idx is None:
            console.clear()
            return 0

        choice = _MENU_CHOICES[choice_idx][0]
        if choice == "explore":
            _explore_dialog(console)
        elif choice == "solve":
            _solve_dialog(console)
        elif choice == "install":
            _install_dialog(console)
        elif choice == "share":
            _share_dialog(console)
        elif choice == "onboarding":
            _onboarding_dialog(console)
        elif choice == "settings" and _settings_dialog(console):
            console.clear()
            return 0


def main() -> int:
    return launch_menu()


def _banner_panel() -> Panel:
    username = _current_username()
    signed_in = f"Signed in as [bold green]{username}[/bold green]" if username else "Not signed in"
    banner = (
        "[bold]Rastion v0.1 - Your Optimization Lab[/bold]\n\n"
        "Pure Rich terminal UI with keyboard navigation.\n"
        f"{signed_in}\n"
        "Use UP/DOWN (or j/k) to move, Enter to select, Q to cancel."
    )
    return Panel(banner, border_style="cyan")


def _view_run_history(console: Console) -> None:
    console.clear()
    history_file = runs_file()
    if not history_file.exists():
        console.print("[yellow]No run history found.[/yellow]")
        wait_for_key(console, title="Run History", prompt="No run history found.")
        return

    lines = [line for line in history_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        console.print("[yellow]No run history found.[/yellow]")
        wait_for_key(console, title="Run History", prompt="No run history found.")
        return

    table = Table(title="Recent Runs")
    table.add_column("Timestamp")
    table.add_column("Solver", style="cyan")
    table.add_column("Status")
    table.add_column("Objective")

    for line in lines[-10:]:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        solution = record.get("solution", {})
        table.add_row(
            str(record.get("timestamp", "")),
            str(record.get("solver_name", "")),
            str(solution.get("status", "")),
            "" if solution.get("objective_value") is None else str(solution.get("objective_value")),
        )

    console.print(table)
    wait_for_key(console, title="Run History", prompt="Recent runs loaded.")


def _show_settings(console: Console) -> None:
    console.clear()
    hub = hub_config()
    table = Table(title="Settings")
    table.add_column("Key", style="green")
    table.add_column("Value", style="white")
    table.add_row("RASTION_HOME", str(rastion_home()))
    table.add_row("config", str(config_file()))
    table.add_row("runs", str(runs_file()))
    table.add_row("hub.url", str(hub.get("url") or ""))
    table.add_row("hub.token", "set" if hub.get("token") else "not set")
    table.add_row("hub.username", _current_username() or "not set")
    console.print(table)
    wait_for_key(console, title="Settings")


def _explore_dialog(console: Console) -> None:
    choice = select_option(
        console,
        title="Explore",
        prompt="Choose what to browse:",
        options=["Browse Problems", "Browse Solvers"],
    )
    if choice is None:
        return
    if choice == 0:
        browse_problems(console)
    else:
        browse_solvers(console)


def _solve_dialog(console: Console) -> None:
    choice = select_option(
        console,
        title="Solve",
        prompt="Choose action:",
        options=["Solve a Problem", "Compare Solvers (Benchmark)", "View Run History"],
    )
    if choice is None:
        return
    if choice == 0:
        solve_problem(console)
    elif choice == 1:
        benchmark_solvers(console)
    else:
        _view_run_history(console)


def _install_dialog(console: Console) -> None:
    choice = select_option(
        console,
        title="Install",
        prompt="Choose package type to install:",
        options=["Install Solver", "Install Problem"],
    )
    if choice is None:
        return
    if choice == 0:
        _install_solver_dialog(console)
    else:
        _install_problem_dialog(console)


def _install_solver_dialog(console: Console) -> None:
    source_idx = select_option(
        console,
        title="Install Solver",
        prompt="Choose solver source:",
        options=["From GitHub ZIP URL", "From Hub"],
    )
    if source_idx is None:
        return

    if source_idx == 0:
        _install_solver_from_url_dialog(console)
    else:
        _install_solver_from_hub_dialog(console)


def _install_problem_dialog(console: Console) -> None:
    _install_problem_from_hub_dialog(console)


def _install_solver_from_url_dialog(console: Console) -> None:
    url = prompt_input(
        console,
        title="Install Solver",
        prompt="GitHub ZIP URL",
        default="https://github.com/YOUR_USERNAME/rastion-solver/archive/main.zip",
    )
    if not url:
        return

    name = prompt_input(
        console,
        title="Install Solver",
        prompt="Optional local solver name (blank to auto-detect)",
        default="",
    )
    if name is None:
        return

    try:
        installed_path = install_solver_from_url(url, name=name.strip() or None)
    except Exception as exc:
        console.clear()
        console.print(f"[red]Install failed: {exc}[/red]")
        wait_for_key(console, title="Install Solver", prompt="Solver install failed.")
        return

    console.clear()
    console.print(f"[green]Installed solver at {installed_path}[/green]")
    wait_for_key(console, title="Install Solver", prompt="Solver installed successfully.")


def _install_solver_from_hub_dialog(console: Console) -> None:
    query = prompt_input(
        console,
        title="Install Solver from Hub",
        prompt="Search query (blank lists latest matches)",
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
        wait_for_key(console, title="Install Solver from Hub", prompt="Hub search failed.")
        return

    solvers = results.get("solvers", [])
    if not solvers:
        console.clear()
        console.print("[yellow]No solver matches found on hub.[/yellow]")
        wait_for_key(console, title="Install Solver from Hub", prompt="No solver matches found.")
        return

    options = [_hub_item_label(item) for item in solvers]
    selected_idx = select_option(
        console,
        title="Install Solver from Hub",
        prompt="Select a solver package:",
        options=options,
    )
    if selected_idx is None:
        return

    selected = solvers[selected_idx]
    selected_name = str(selected.get("name") or "")
    if not selected_name:
        console.clear()
        console.print("[red]Selected solver entry is missing a name.[/red]")
        wait_for_key(console, title="Install Solver from Hub", prompt="Install failed.")
        return

    overwrite = _confirm(console, title="Install Solver from Hub", prompt="Overwrite existing local solver if needed?")

    try:
        pulled = asyncio.run(client.pull(selected_name, type="solver", overwrite=overwrite))
    except (HubClientError, OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        console.clear()
        console.print(f"[red]Install from hub failed: {exc}[/red]")
        wait_for_key(console, title="Install Solver from Hub", prompt="Install from hub failed.")
        return

    console.clear()
    console.print(
        f"[green]Installed solver '{pulled.get('name', selected_name)}' at {pulled.get('path', 'unknown path')}[/green]"
    )
    wait_for_key(console, title="Install Solver from Hub", prompt="Solver installed successfully.")


def _install_problem_from_hub_dialog(console: Console) -> None:
    query = prompt_input(
        console,
        title="Install Problem from Hub",
        prompt="Search query (blank lists latest matches)",
        default="",
    )
    if query is None:
        return

    try:
        client = HubClient()
        results = asyncio.run(client.search(query.strip(), type="problem"))
    except (HubClientError, OSError, RuntimeError, ValueError) as exc:
        console.clear()
        console.print(f"[red]Hub search failed: {exc}[/red]")
        wait_for_key(console, title="Install Problem from Hub", prompt="Hub search failed.")
        return

    problems = results.get("problems", [])
    if not problems:
        console.clear()
        console.print("[yellow]No problem matches found on hub.[/yellow]")
        wait_for_key(console, title="Install Problem from Hub", prompt="No problem matches found.")
        return

    options = [_hub_item_label(item) for item in problems]
    selected_idx = select_option(
        console,
        title="Install Problem from Hub",
        prompt="Select a problem package:",
        options=options,
    )
    if selected_idx is None:
        return

    selected = problems[selected_idx]
    selected_name = str(selected.get("name") or "")
    if not selected_name:
        console.clear()
        console.print("[red]Selected problem entry is missing a name.[/red]")
        wait_for_key(console, title="Install Problem from Hub", prompt="Install failed.")
        return

    overwrite = _confirm(console, title="Install Problem from Hub", prompt="Overwrite existing local problem if needed?")

    try:
        pulled = asyncio.run(client.pull(selected_name, type="problem", overwrite=overwrite))
    except (HubClientError, OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        console.clear()
        console.print(f"[red]Install from hub failed: {exc}[/red]")
        wait_for_key(console, title="Install Problem from Hub", prompt="Install from hub failed.")
        return

    console.clear()
    console.print(
        f"[green]Installed problem '{pulled.get('name', selected_name)}' at {pulled.get('path', 'unknown path')}[/green]"
    )
    wait_for_key(console, title="Install Problem from Hub", prompt="Problem installed successfully.")


def _share_dialog(console: Console) -> None:
    choice = select_option(
        console,
        title="Share",
        prompt="Choose hub action:",
        options=["Hub Push", "Hub Pull", "Hub Search"],
    )
    if choice is None:
        return
    if choice == 0:
        _hub_push_dialog(console)
    elif choice == 1:
        _hub_pull_dialog(console)
    else:
        _hub_search_dialog(console)


def _hub_push_dialog(console: Console) -> None:
    package_path = prompt_input(
        console,
        title="Hub Push",
        prompt="Path to local problem or solver folder",
        default=str(Path.cwd()),
    )
    if not package_path:
        return

    type_idx = select_option(
        console,
        title="Hub Push",
        prompt="Package type:",
        options=["Auto-detect", "Problem", "Solver"],
    )
    if type_idx is None:
        return

    package_type: str | None
    if type_idx == 1:
        package_type = "problem"
    elif type_idx == 2:
        package_type = "solver"
    else:
        package_type = None

    try:
        client = HubClient()
        result = asyncio.run(client.push(Path(package_path).expanduser(), type=package_type))
    except (HubClientError, OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        if _is_hub_auth_error(exc):
            should_login = _confirm(
                console,
                title="Hub Push",
                prompt="Push requires authentication. Start onboarding now?",
            )
            if should_login:
                _login_with_github(console)
            return
        console.clear()
        console.print(f"[red]Push failed: {exc}[/red]")
        wait_for_key(console, title="Hub Push", prompt="Hub push failed.")
        return

    pkg_type = str(result.get("type") or package_type or "package")
    pkg_name = str(result.get("name") or Path(package_path).name)
    console.clear()
    console.print(f"[green]Pushed {pkg_type} '{pkg_name}' to {client.base_url}[/green]")
    wait_for_key(console, title="Hub Push", prompt="Hub push completed.")


def _hub_pull_dialog(console: Console) -> None:
    name = prompt_input(
        console,
        title="Hub Pull",
        prompt="Package name",
        default="",
    )
    if not name:
        return

    type_idx = select_option(
        console,
        title="Hub Pull",
        prompt="Package type:",
        options=["Both", "Problem", "Solver"],
    )
    if type_idx is None:
        return

    pkg_type = "both"
    if type_idx == 1:
        pkg_type = "problem"
    elif type_idx == 2:
        pkg_type = "solver"

    overwrite = _confirm(console, title="Hub Pull", prompt="Overwrite local package if it exists?")

    try:
        client = HubClient()
        result = asyncio.run(client.pull(name, type=pkg_type, overwrite=overwrite))
    except (HubClientError, OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        console.clear()
        console.print(f"[red]Pull failed: {exc}[/red]")
        wait_for_key(console, title="Hub Pull", prompt="Hub pull failed.")
        return

    pulled_name = str(result.get("name") or name)
    pulled_type = str(result.get("type") or pkg_type)
    pulled_path = str(result.get("path") or "")
    console.clear()
    console.print(f"[green]Pulled {pulled_type} '{pulled_name}' into {pulled_path}[/green]")
    wait_for_key(console, title="Hub Pull", prompt="Hub pull completed.")


def _hub_search_dialog(console: Console) -> None:
    query = prompt_input(
        console,
        title="Hub Search",
        prompt="Search query",
        default="",
    )
    if query is None:
        return

    type_idx = select_option(
        console,
        title="Hub Search",
        prompt="Search scope:",
        options=["Both", "Problems", "Solvers"],
    )
    if type_idx is None:
        return

    search_type = "both"
    if type_idx == 1:
        search_type = "problem"
    elif type_idx == 2:
        search_type = "solver"

    try:
        client = HubClient()
        results = asyncio.run(client.search(query.strip(), type=search_type))
    except (HubClientError, OSError, RuntimeError, ValueError) as exc:
        console.clear()
        console.print(f"[red]Search failed: {exc}[/red]")
        wait_for_key(console, title="Hub Search", prompt="Hub search failed.")
        return

    _show_hub_search_results(console, results)


def _show_hub_search_results(console: Console, results: dict[str, list[dict[str, Any]]]) -> None:
    console.clear()
    problems = results.get("problems", [])
    solvers = results.get("solvers", [])

    if not problems and not solvers:
        console.print("[yellow]No hub matches found.[/yellow]")
        wait_for_key(console, title="Hub Search", prompt="No hub matches found.")
        return

    if problems:
        table = Table(title="Hub Problems")
        table.add_column("Name", style="cyan")
        table.add_column("Version")
        table.add_column("Owner")
        table.add_column("Downloads")
        table.add_column("Rating")
        for item in problems:
            table.add_row(
                str(item.get("name", "")),
                str(item.get("version", "")),
                _hub_owner(item),
                str(item.get("download_count", "n/a")),
                str(item.get("rating", "n/a")),
            )
        console.print(table)

    if solvers:
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
        console.print(table)

    wait_for_key(console, title="Hub Search", prompt="Hub search results loaded.")


def _onboarding_dialog(console: Console) -> None:
    choice = select_option(
        console,
        title="Onboarding",
        prompt="Choose action:",
        options=["Start onboarding with GitHub token", "Logout"],
    )
    if choice is None:
        return
    if choice == 0:
        _login_with_github(console)
    else:
        _logout(console)


def _settings_dialog(console: Console) -> bool:
    choice = select_option(
        console,
        title="Settings",
        prompt="Choose action:",
        options=["View Settings", "Quit Rastion"],
    )
    if choice is None:
        return False
    if choice == 0:
        _show_settings(console)
        return False
    return _confirm(console, title="Quit Rastion", prompt="Exit Rastion now?")


def _login_with_github(console: Console) -> None:
    entered_token = prompt_input(
        console,
        title="Onboarding",
        prompt="GitHub token",
        default="",
    )
    if entered_token is None:
        return

    token = entered_token.strip()
    using_env_token = False
    if not token:
        token = _env_github_token() or ""
        using_env_token = bool(token)

    if not token:
        console.clear()
        console.print("[red]Token cannot be empty and no token was found in local .env or environment.[/red]")
        wait_for_key(console, title="Onboarding", prompt="Login failed.")
        return

    try:
        client = HubClient()
        user = asyncio.run(client.login(token))
        username = _hub_username_from_user_payload(user)
        _save_login(token=client.token, username=username)
    except (HubClientError, OSError, RuntimeError, ValueError) as exc:
        console.clear()
        console.print(f"[red]Login failed: {exc}[/red]")
        wait_for_key(console, title="Onboarding", prompt="Login failed.")
        return

    phrase = random.choice(_ONBOARDING_PHRASES)
    console.clear()
    if using_env_token:
        console.print("[cyan]Using GitHub token from local .env/environment.[/cyan]")
    console.print(
        Panel(
            f"[bold green]welcome {username}[/bold green]\n\n{phrase}",
            title="Onboarding complete",
            border_style="green",
        )
    )
    console.print(f"Token saved to {config_file()}")
    wait_for_key(console, title="Onboarding", prompt="Onboarding successful.")


def _logout(console: Console) -> None:
    previous = _current_username()
    _save_login(token=None, username=None)
    console.clear()
    if previous:
        console.print(f"[green]Logged out {previous}.[/green]")
    else:
        console.print("[green]Logged out.[/green]")
    wait_for_key(console, title="Logout", prompt="Credentials cleared.")


def _save_login(*, token: str | None, username: str | None) -> None:
    cfg = load_config()
    hub = cfg.get("hub")
    if not isinstance(hub, dict):
        hub = {}
    hub["token"] = token
    hub["username"] = username
    cfg["hub"] = hub
    write_config(cfg)


def _current_username() -> str | None:
    cfg = load_config()
    hub = cfg.get("hub")
    if not isinstance(hub, dict):
        return None
    token = hub.get("token")
    username = hub.get("username")
    if not token:
        return None
    if isinstance(username, str) and username.strip():
        return username.strip()
    return None


def _hub_username_from_user_payload(user: dict[str, Any]) -> str:
    username = user.get("username")
    if isinstance(username, str) and username.strip():
        return username.strip()

    login = user.get("login")
    if isinstance(login, str) and login.strip():
        return login.strip()

    raise RuntimeError("Hub login response did not include a valid username.")


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


def _confirm(console: Console, *, title: str, prompt: str) -> bool:
    idx = select_option(
        console,
        title=title,
        prompt=prompt,
        options=["No", "Yes"],
    )
    if idx is None:
        return False
    return idx == 1


def _is_hub_auth_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    return "not authenticated" in message or "unauthorized" in message or "401" in message


def _hub_item_label(item: dict[str, Any]) -> str:
    name = str(item.get("name", "unknown"))
    version = str(item.get("version", "n/a"))
    owner = _hub_owner(item)
    return f"{name} v{version} ({owner})"


def _hub_owner(item: dict[str, Any]) -> str:
    owner = item.get("owner")
    if isinstance(owner, dict):
        username = owner.get("username")
        if isinstance(username, str) and username.strip():
            return username
    return "unknown"
