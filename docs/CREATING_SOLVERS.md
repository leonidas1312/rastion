# Creating Solvers in Rastion

This guide shows how to create and share a custom solver plugin.

## Minimal Contract

A solver plugin archive must include a `solver.py` file with:

- a class inheriting `SolverPlugin`
- `name` and `version`
- `capabilities()`
- `solve(ir_model, config, backend)`
- `get_plugin()` factory

Use the template:

- `rastion/solvers/solver_template.py`
- `rastion/solvers/solver_card_template.md`

## 1) Start From the Template

Copy `rastion/solvers/solver_template.py` into your own repo as `solver.py`.

Edit:

- `MySolverPlugin` class name
- `name` (unique solver id)
- `version`
- capability declaration
- `solve()` implementation

## 2) Implement `get_plugin()`

Return `None` if dependencies are missing so discovery can skip cleanly.

## 3) Optional: Add `solver_card.md`

This is not required for loading, but useful as package documentation.

## 4) Publish on GitHub

Put `solver.py` at repo root and share a ZIP URL, for example:

```text
https://github.com/YOUR_USERNAME/rastion-solver-yourname/archive/main.zip
```

## 5) Install Locally

```bash
rastion install-solver https://github.com/YOUR_USERNAME/rastion-solver-yourname/archive/main.zip
```

Optional custom folder name:

```bash
rastion install-solver <URL> --name my_solver_repo
```

## 6) Verify Discovery

```bash
rastion info
```

Or open TUI solver browser:

```bash
rastion
```

## 7) Benchmark and History

Installed external solvers participate in:

- Solve flow
- Benchmark flow
- Run history logging
