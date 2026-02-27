# Facility Location (MILP)

Choose which facilities to open and assign customers while respecting capacities.

- Variables: binary open decisions + continuous assignments
- Objective: fixed opening + assignment costs
- Constraints: one assignment per customer and capacity limits

Validate:

```bash
python -m rastion.cli.main validate examples/facility_location/spec.json examples/facility_location/instance.json
```

Solve (if HiGHS or OR-Tools installed):

```bash
python -m rastion.cli.main solve examples/facility_location/spec.json examples/facility_location/instance.json --solver auto
```
