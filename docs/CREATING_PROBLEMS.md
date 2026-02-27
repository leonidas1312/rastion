# Creating Problems in Rastion

This guide shows how to package a portable optimization problem for Rastion.

## Folder Layout

A shareable problem folder should look like:

```text
my_problem/
  spec.json
  instances/
    default.json
  metadata.yaml           # optional
  problem_card.md         # optional
```

Legacy `instance.json` is still supported, but `instances/default.json` is recommended.

## 1) Write `spec.json`

`spec.json` defines model structure:

- variables (`binary`, `integer`, `continuous`)
- objective references (`linear`, optional `quadratic`)
- constraints (linear blocks)
- `ir_target` (`generic` or `qubo`)

Use existing examples in `examples/` as references.

## 2) Write Instance Data

Create `instances/default.json` with arrays/parameters referenced by `spec.json`.

For large numeric payloads, NPZ instances are also supported (`instances/default.npz`).

## 3) Add Metadata (Optional)

`metadata.yaml` improves registry browsing:

```yaml
name: my_problem
version: 0.1.0
author: your_name
tags: [milp, scheduling]
optimization_class: MILP
difficulty: medium
```

## 4) Add a Problem Card (Optional)

`problem_card.md` is rendered in the TUI and should briefly describe:

- objective
- variables/constraints
- recommended solvers

## 5) Validate

```bash
rastion validate my_problem/spec.json my_problem/instances/default.json
```

## 6) Install into Local Registry

```bash
rastion install ./my_problem
```

Then browse or solve from the TUI:

```bash
rastion
```

## 7) Export for Sharing

```bash
rastion export my_problem ./shareable_my_problem
```

Another user can install with:

```bash
rastion install ./shareable_my_problem
```
