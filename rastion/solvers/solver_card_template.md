# My Solver

Brief description of what this solver does.

## Capabilities

- Variables: binary, integer, continuous
- Objectives: linear, quadratic, QUBO
- Constraints: linear

## When to Use

Best for [describe use cases].

## Installation

```bash
rastion install-solver https://github.com/YOUR_USERNAME/rastion-solver-YOURNAME/archive/main.zip
```

## Example

```python
from rastion import Problem, AutoSolver
from rastion.compile.normalize import compile_to_ir

problem = Problem.from_registry("my_problem")
solver = AutoSolver.from_preferences(preferred=["my_solver"])
ir_model = compile_to_ir(problem.spec, problem.load_instance("default"))
result = solver.plugin.solve(ir_model, config={}, backend=None)
```
