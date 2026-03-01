"""Scaffolding utilities for agent-first Rastion workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def init_agent_workspace(path: str | Path = ".", *, overwrite: bool = False) -> list[Path]:
    """Create a local agent bootstrap with policy and decision template files."""
    root = Path(path).expanduser().resolve()
    files: dict[Path, str] = {root / "AGENTS.md": _agents_markdown()}

    files.update(
        _decision_bundle_files(
            root=root,
            name="calendar_week_off",
            spec=_calendar_spec(),
            instance=_calendar_instance(),
            metadata=_metadata_yaml(name="calendar_week_off", description="Starter half-day calendar optimizer."),
            card=_problem_card_markdown_half_day(),
            decision_yaml=_decision_yaml(name="calendar_week_off"),
        )
    )

    spec_30m, instance_30m = _calendar_30m_problem()
    files.update(
        _decision_bundle_files(
            root=root,
            name="calendar_week_off_30m",
            spec=spec_30m,
            instance=instance_30m,
            metadata=_metadata_yaml(
                name="calendar_week_off_30m",
                description="30-minute calendar optimizer with fixed meeting blackout support.",
            ),
            card=_problem_card_markdown_30m(),
            decision_yaml=_decision_yaml(name="calendar_week_off_30m"),
        )
    )

    files[root / "agent_requests" / "calendar_week_off.json"] = _json_pretty(
        _agent_request_template(decision_plugin_path="../decision_plugins/calendar_week_off")
    )
    files[root / "agent_requests" / "calendar_week_off_30m.json"] = _json_pretty(
        _agent_request_template(decision_plugin_path="../decision_plugins/calendar_week_off_30m")
    )
    files[root / "agent_requests" / "calendar_week_off_30m_remote_solver.json"] = _json_pretty(
        _agent_request_template(
            decision_plugin_path="../decision_plugins/calendar_week_off_30m",
            allow_remote_install=True,
            install_urls=["https://github.com/YOUR_ORG/YOUR_SOLVER/archive/refs/heads/main.zip"],
        )
    )

    existing = [target for target in files if target.exists()]
    if existing and not overwrite:
        sample = ", ".join(str(path.relative_to(root)) for path in existing[:3])
        more = "" if len(existing) <= 3 else f" (+{len(existing) - 3} more)"
        raise FileExistsError(
            "agent workspace already contains bootstrap files: "
            + sample
            + more
            + " (use overwrite=True to replace)"
        )

    for target, content in files.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    return sorted(files)


def _json_pretty(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _decision_bundle_files(
    *,
    root: Path,
    name: str,
    spec: dict[str, object],
    instance: dict[str, object],
    metadata: str,
    card: str,
    decision_yaml: str,
) -> dict[Path, str]:
    decision_root = root / "decision_plugins" / name
    return {
        decision_root / "decision.yaml": decision_yaml,
        decision_root / "spec.json": _json_pretty(spec),
        decision_root / "instances" / "default.json": _json_pretty(instance),
        decision_root / "metadata.yaml": metadata,
        decision_root / "problem_card.md": card,
    }


def _calendar_spec() -> dict[str, object]:
    return {
        "schema_version": "0.1.0",
        "name": "calendar_week_off",
        "ir_target": "generic",
        "variables": [
            {"name": "x_mon_am", "vartype": "binary"},
            {"name": "x_mon_pm", "vartype": "binary"},
            {"name": "x_tue_am", "vartype": "binary"},
            {"name": "x_tue_pm", "vartype": "binary"},
            {"name": "x_wed_am", "vartype": "binary"},
            {"name": "x_wed_pm", "vartype": "binary"},
            {"name": "x_thu_am", "vartype": "binary"},
            {"name": "x_thu_pm", "vartype": "binary"},
            {"name": "x_fri_am", "vartype": "binary"},
            {"name": "x_fri_pm", "vartype": "binary"},
        ],
        "objective": {
            "sense": "max",
            "linear": "utility",
            "constant": 0.0,
        },
        "constraints": [
            {
                "name": "daily_limit",
                "matrix": "A_daily",
                "rhs": "b_daily",
                "sense": "<=",
            },
            {
                "name": "friday_off",
                "matrix": "A_friday_off",
                "rhs": "b_friday_off",
                "sense": "<=",
            },
        ],
    }


def _calendar_instance() -> dict[str, object]:
    return {
        "schema_version": "0.1.0",
        "arrays": {
            "utility": [9, 6, 8, 7, 7, 6, 8, 7, 1, 1],
            "A_daily": [
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            ],
            "b_daily": [1, 1, 1, 1, 1],
            "A_friday_off": [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]],
            "b_friday_off": [0],
        },
        "params": {
            "description": "Agent starter template. Replace utility/constraints with user calendar data.",
            "horizon": "one_week",
            "slot_size": "half_day",
        },
    }


def _agents_markdown() -> str:
    return """# AGENTS.md

## Rastion Agent Contract
- Treat `rastion` as a local optimization runtime.
- Keep model/API keys out of `rastion` solve flows.
- Use Hub login only for optional marketplace actions.

## Default Workflow
1. Ask for goal, hard constraints, soft preferences, horizon, and timezone.
2. Convert request to a local decision plugin folder with `spec.json` and `instances/default.json`.
3. Prefer `rastion agent-run` with a request JSON under `agent_requests/`.
4. If needed, run direct:
   - `rastion plugin install ./<decision-plugin-folder> --overwrite`
   - `rastion info --verbose <plugin-name> default`
   - `rastion solve <plugin-name> default --solver auto --time-limit 30`
5. Explain output in plain language, including tradeoffs and unmet soft preferences.

## Solver Policy
- Prefer local installed solvers first.
- If no compatible solver exists, suggest:
  - `pip install "rastion[highs]"` (first choice)
  - `pip install "rastion[ortools]"`
- Only install remote solver code when the user explicitly approves source trust.

## Calendar Prompt Template
"Convert this calendar into an optimization model. Hard constraints: <...>. Soft preferences: <...>.
Return an optimized weekly schedule and explain tradeoffs."
"""


def _decision_yaml(*, name: str) -> str:
    return f"""schema_version: "0.1.0"
name: "{name}"
version: "0.1.0"
type: "decision_plugin"
description: "Starter bundle for agent-led weekly calendar optimization."
decision_plugin_path: "."
preferred_solvers:
  - highs
  - ortools
fallback:
  allow_remote_solver_install: false
  require_user_confirmation: true
"""


def _metadata_yaml(*, name: str, description: str) -> str:
    return f"""name: {name}
version: 0.1.0
author: local
tags:
  - scheduling
  - calendar
  - agentic
optimization_class: MILP
difficulty: easy
description: "{description}"
"""


def _problem_card_markdown_half_day() -> str:
    return """# calendar_week_off

Starter decision template for "keep Friday off" weekly planning.

## What This Models
- One binary decision variable per half-day slot.
- Objective maximizes `utility`.
- `daily_limit` enforces at most one focus block per day.
- `friday_off` enforces zero Friday blocks.

## How Agents Should Adapt It
1. Expand slots to hourly or 30-minute granularity.
2. Convert fixed meetings into equality or exclusion constraints.
3. Add objective penalties for context-switching and late-day work.
4. Keep hard constraints separate from soft preferences via weighted terms.
"""


def _problem_card_markdown_30m() -> str:
    return """# calendar_week_off_30m

30-minute decision template with fixed-meeting blackout and Friday-off constraints.

## What This Models
- 80 binary decision variables: 5 weekdays x 16 half-hour slots (09:00-17:00).
- `weekly_target` enforces an exact number of focused work slots.
- `meeting_blackout` blocks slots occupied by fixed meetings.
- `daily_cap` limits overload per day.
- `friday_off` enforces no focus slots on Friday.

## How Agents Should Adapt It
1. Expand weekday set or working hours from user input.
2. Replace sample fixed meetings using the user's real calendar events.
3. Tune `utility` weights to reflect energy and preference profiles.
4. Add constraint blocks for lunch windows, commute windows, or context switches.
"""


def _agent_request_template(
    *,
    decision_plugin_path: str,
    allow_remote_install: bool = False,
    install_urls: list[str] | None = None,
) -> dict[str, object]:
    return {
        "decision_plugin_path": decision_plugin_path,
        "instance_name": "default",
        "install_overwrite": True,
        "solver": {
            "name": "auto",
            "preferred": ["highs", "ortools"],
            "config": {
                "time_limit": 30,
            },
        },
        "solver_policy": {
            "allow_remote_install": allow_remote_install,
            "install_urls": install_urls or [],
            "overwrite": False,
        },
        "runs_dir": "../.rastion-home/runs",
        "output_json": "../.rastion-home/runs/last_agent_run.json",
    }


def _calendar_30m_problem() -> tuple[dict[str, object], dict[str, object]]:
    slots = _calendar_30m_slots()
    names = [slot["name"] for slot in slots]
    n = len(names)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    day_indices: dict[str, list[int]] = {}
    for idx, slot in enumerate(slots):
        day_indices.setdefault(str(slot["day"]), []).append(idx)

    fixed_meeting_slots = [
        "x_tue_10_00",
        "x_tue_10_30",
        "x_wed_14_00",
        "x_wed_14_30",
        "x_thu_11_30",
        "x_thu_12_00",
    ]
    fixed_indices = [name_to_idx[name] for name in fixed_meeting_slots]

    utility = [_slot_utility(slot) for slot in slots]
    a_weekly_target = [[1.0] * n]
    b_weekly_target = [24.0]
    a_daily = []
    for day in ["mon", "tue", "wed", "thu", "fri"]:
        row = [0.0] * n
        for idx in day_indices[day]:
            row[idx] = 1.0
        a_daily.append(row)
    b_daily = [6.0, 6.0, 6.0, 6.0, 6.0]

    friday_row = [0.0] * n
    for idx in day_indices["fri"]:
        friday_row[idx] = 1.0

    a_meeting = []
    for idx in fixed_indices:
        row = [0.0] * n
        row[idx] = 1.0
        a_meeting.append(row)

    spec = {
        "schema_version": "0.1.0",
        "name": "calendar_week_off_30m",
        "ir_target": "generic",
        "variables": [{"name": name, "vartype": "binary"} for name in names],
        "objective": {"sense": "max", "linear": "utility", "constant": 0.0},
        "constraints": [
            {"name": "weekly_target", "matrix": "A_weekly_target", "rhs": "b_weekly_target", "sense": "=="},
            {"name": "daily_cap", "matrix": "A_daily", "rhs": "b_daily", "sense": "<="},
            {"name": "meeting_blackout", "matrix": "A_meeting_blackout", "rhs": "b_meeting_blackout", "sense": "<="},
            {"name": "friday_off", "matrix": "A_friday_off", "rhs": "b_friday_off", "sense": "<="},
        ],
    }

    instance = {
        "schema_version": "0.1.0",
        "arrays": {
            "utility": utility,
            "A_weekly_target": a_weekly_target,
            "b_weekly_target": b_weekly_target,
            "A_daily": a_daily,
            "b_daily": b_daily,
            "A_meeting_blackout": a_meeting,
            "b_meeting_blackout": [0.0] * len(a_meeting),
            "A_friday_off": [friday_row],
            "b_friday_off": [0.0],
        },
        "params": {
            "slot_minutes": 30,
            "timezone": "local",
            "horizon": "one_week_weekdays",
            "fixed_meeting_slots": fixed_meeting_slots,
            "slot_labels": [slot["label"] for slot in slots],
        },
    }
    return spec, instance


def _calendar_30m_slots() -> list[dict[str, str | int]]:
    slots: list[dict[str, str | int]] = []
    for day_code, day_name in [
        ("mon", "Monday"),
        ("tue", "Tuesday"),
        ("wed", "Wednesday"),
        ("thu", "Thursday"),
        ("fri", "Friday"),
    ]:
        for hour in range(9, 17):
            for minute in (0, 30):
                slots.append(
                    {
                        "day": day_code,
                        "hour": hour,
                        "minute": minute,
                        "name": f"x_{day_code}_{hour:02d}_{minute:02d}",
                        "label": f"{day_name} {hour:02d}:{minute:02d}",
                    }
                )
    return slots


def _slot_utility(slot: dict[str, Any]) -> float:
    day = str(slot["day"])
    hour = int(slot["hour"])
    minute = int(slot["minute"])

    day_weight = {"mon": 8.0, "tue": 7.5, "wed": 7.0, "thu": 7.2, "fri": 1.5}[day]
    if hour < 12:
        time_bonus = 2.2
    elif hour < 15:
        time_bonus = 1.2
    else:
        time_bonus = 0.4
    minute_adjust = -0.1 if minute == 30 else 0.0
    return round(day_weight + time_bonus + minute_adjust, 3)
