# Traveling Salesman (MILP, MTZ)

4-city TSP encoded as a MILP with MTZ subtour elimination constraints.

- Variables: binary arc decisions and continuous order variables
- Objective: minimize travel distance
- Constraints: one incoming/outgoing edge per city + MTZ subtour elimination

Validate:

```bash
python -m rastion.cli.main validate examples/tsp/spec.json examples/tsp/instance.json
```

Solve (if OR-Tools/HiGHS installed):

```bash
python -m rastion.cli.main solve examples/tsp/spec.json examples/tsp/instance.json --solver auto
```
