<p align="center">
  <img src="assets/owl-favicon.png" alt="Rastion owl logo" width="96" />
</p>

# Rastion

Local-first optimization lab for decision plugins.

## MVP Scope

- Core MVP is local-first: model, run, and inspect decision plugins on your own machine.
- Hub/community features are optional and can be added after local workflow is stable.
- User-facing artifact is the decision plugin (`spec.json` + `instances/` + metadata/card).

[![PyPI version](https://img.shields.io/pypi/v/rastion.svg)](https://pypi.org/project/rastion/)
[![Python versions](https://img.shields.io/pypi/pyversions/rastion.svg)](https://pypi.org/project/rastion/)
[![License](https://img.shields.io/pypi/l/rastion.svg)](LICENSE)

## Install

Latest release:

```bash
pip install rastion
```

Run the app:

```bash
rastion
```

## Agent-First Quickstart

Scaffold a Codex-friendly workspace:

```bash
rastion init-agent --path .
```

This creates:
- `AGENTS.md` with a local-first optimization workflow policy.
- `decision_plugins/calendar_week_off/` starter files (half-day model).
- `decision_plugins/calendar_week_off_30m/` starter files (30-minute slots + fixed meeting blackout).
- `agent_requests/*.json` structured requests for `rastion agent-run`.

Then run the starter template:

```bash
export RASTION_HOME=.rastion-home
rastion init-registry
rastion agent-run ./agent_requests/calendar_week_off_30m.json
```

To allow agent-orchestrated solver download fallback (trusted URL list in request JSON):

```bash
rastion agent-run ./agent_requests/calendar_week_off_30m_remote_solver.json
```

## Talk To Your Optimization Lab

Use this prompt shape with your coding agent:

```text
Convert this request into a Rastion decision plugin.
Goal: <what outcome you want>
Hard constraints: <must hold>
Soft preferences: <nice to have>
Horizon: <time window>
Timezone: <timezone>
Return: decision_plugins/<name>/spec.json, instances/default.json, and an agent_request JSON.
```

Then run:

```bash
rastion plugin install ./decision_plugins/<name> --overwrite
rastion info --verbose <name> default
rastion solve <name> default --solver auto --time-limit 30
```

Inspect state deterministically:

```bash
rastion plugin list --json
rastion plugin dev-list --path ./decision_plugins --json
rastion plugin status <name> --path ./decision_plugins --json
rastion runs list --json
```

## Decision Plugin Workflow

Install a local plugin folder into your registry:

```bash
rastion plugin install ./decision_plugins/calendar_week_off_30m --overwrite
```

List installed plugins:

```bash
rastion plugin list
```

List local workspace plugins (not yet installed):

```bash
rastion plugin dev-list --path ./decision_plugins
```

View run history:

```bash
rastion runs list
```

Push a plugin to Rastion Hub:

```bash
rastion login
rastion plugin push ./decision_plugins/calendar_week_off_30m
```

Search and pull from Rastion Hub:

```bash
rastion plugin search calendar
rastion plugin pull calendar_week_off_30m --overwrite
```
