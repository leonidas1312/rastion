# Portfolio Optimization (QP)

Minimize variance minus expected return for 3 assets.

- Variables: continuous weights in [0,1]
- Objective: quadratic covariance term + linear return penalty
- Constraints: full budget and minimum expected return

Validate:

```bash
python -m rastion.cli.main validate examples/portfolio/spec.json examples/portfolio/instance.json
```

Solve (if HiGHS installed):

```bash
python -m rastion.cli.main solve examples/portfolio/spec.json examples/portfolio/instance.json --solver highs
```
