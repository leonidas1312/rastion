# calendar_week_off

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
