# calendar_week_off_30m

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
