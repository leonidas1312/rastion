# AGENTS.md

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
