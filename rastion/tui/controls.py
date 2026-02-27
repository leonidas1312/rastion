"""Shared Rich + readchar controls for in-console TUI interaction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import readchar
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

_ENTER_KEYS = {readchar.key.ENTER, "\r", "\n"}
_UP_KEYS = {readchar.key.UP, "k", "K"}
_DOWN_KEYS = {readchar.key.DOWN, "j", "J"}
_CANCEL_KEYS = {"q", "Q"}


def _compose_screen(
    panel: Panel,
    extra_renderables: Sequence[RenderableType] | None = None,
) -> RenderableType:
    if not extra_renderables:
        return panel
    return Group(*extra_renderables, Text(), panel)


def _render_select_panel(prompt: str, options: Sequence[str], selected: int, allow_cancel: bool) -> Text:
    body = Text()
    body.append(f"{prompt}\n\n", style="bold")
    for idx, option in enumerate(options):
        marker = "▶ " if idx == selected else "  "
        style = "bold cyan" if idx == selected else "white"
        body.append(f"{marker}{option}\n", style=style)
    hint = "↑/↓ or j/k navigate  Enter select"
    if allow_cancel:
        hint += "  Q cancel"
    body.append(f"\n{hint}", style="dim")
    return body


def _render_multiselect_panel(
    prompt: str,
    options: Sequence[str],
    cursor: int,
    chosen: set[int],
    allow_cancel: bool,
) -> Text:
    body = Text()
    body.append(f"{prompt}\n\n", style="bold")
    for idx, option in enumerate(options):
        marker = "▶ " if idx == cursor else "  "
        checked = "[x]" if idx in chosen else "[ ]"
        style = "bold cyan" if idx == cursor else "white"
        body.append(f"{marker}{checked} {option}\n", style=style)
    hint = "↑/↓ or j/k move  Space toggle  Enter confirm"
    if allow_cancel:
        hint += "  Q cancel"
    body.append(f"\n{hint}", style="dim")
    return body


def select_option(
    console: Console,
    *,
    title: str,
    prompt: str,
    options: Sequence[str],
    allow_cancel: bool = True,
    extra_renderables: Sequence[RenderableType] | None = None,
) -> int | None:
    if not options:
        return None

    selected = 0
    with Live(console=console, screen=True, auto_refresh=False, transient=True) as live:
        while True:
            live.update(
                _compose_screen(
                    Panel(
                        _render_select_panel(prompt, options, selected, allow_cancel),
                        title=title,
                        border_style="cyan",
                    ),
                    extra_renderables=extra_renderables,
                ),
                refresh=True,
            )

            key = readchar.readkey()
            if key in _UP_KEYS:
                selected = (selected - 1) % len(options)
                continue
            if key in _DOWN_KEYS:
                selected = (selected + 1) % len(options)
                continue
            if key in _ENTER_KEYS:
                return selected
            if allow_cancel and key in _CANCEL_KEYS:
                return None


def select_multiple(
    console: Console,
    *,
    title: str,
    prompt: str,
    options: Sequence[str],
    allow_cancel: bool = True,
    extra_renderables: Sequence[RenderableType] | None = None,
) -> list[int] | None:
    if not options:
        return []

    cursor = 0
    chosen: set[int] = set()

    with Live(console=console, screen=True, auto_refresh=False, transient=True) as live:
        while True:
            live.update(
                _compose_screen(
                    Panel(
                        _render_multiselect_panel(prompt, options, cursor, chosen, allow_cancel),
                        title=title,
                        border_style="cyan",
                    ),
                    extra_renderables=extra_renderables,
                ),
                refresh=True,
            )

            key = readchar.readkey()
            if key in _UP_KEYS:
                cursor = (cursor - 1) % len(options)
                continue
            if key in _DOWN_KEYS:
                cursor = (cursor + 1) % len(options)
                continue
            if key == " ":
                if cursor in chosen:
                    chosen.remove(cursor)
                else:
                    chosen.add(cursor)
                continue
            if key in _ENTER_KEYS:
                return sorted(chosen)
            if allow_cancel and key in _CANCEL_KEYS:
                return None


def prompt_input(
    console: Console,
    *,
    title: str,
    prompt: str,
    default: str = "",
    allow_cancel: bool = True,
) -> str | None:
    console.clear()
    panel = Text()
    panel.append(f"{prompt}\n\n", style="bold")
    hint = "Type your value and press Enter."
    if allow_cancel:
        hint += " Type Q to cancel."
    panel.append(hint, style="dim")
    console.print(Panel(panel, title=title, border_style="cyan"))
    answer = cast(str, Prompt.ask(f"[cyan]{prompt}[/cyan]", default=default, console=console))
    if allow_cancel and answer.strip().lower() == "q":
        return None
    return answer


def wait_for_key(console: Console, *, title: str, prompt: str = "Press any key to continue.") -> None:
    text = Text()
    text.append(f"{prompt}\n\n", style="bold")
    text.append("Press any key to continue...", style="dim")
    console.print(Panel(text, title=title, border_style="cyan"))
    readchar.readkey()
