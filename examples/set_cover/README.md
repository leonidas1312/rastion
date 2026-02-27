# Set Cover (Binary MILP)

Select a minimum-cost subset of sets that covers all elements.

- Variables: binary set-selection decisions
- Objective: minimize set cost
- Constraints: each element must be covered by at least one chosen set

Validate:

```bash
python -m rastion.cli.main validate examples/set_cover/spec.json examples/set_cover/instance.json
```

Solve (baseline works offline):

```bash
python -m rastion.cli.main solve examples/set_cover/spec.json examples/set_cover/instance.json --solver auto
```
