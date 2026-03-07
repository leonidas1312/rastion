# Contributing

Rastion `v0.1` is a public TSP hub. Contributions should preserve that contract: executable adapters, explicit
metadata, and reproducible suite outputs.

## Add A Solver

1. Add an adapter under `plugins_local/`.
2. Add a solver card at `catalog/solvers/<solver-id>/card.json`.
3. Add `catalog/solvers/<solver-id>/README.md`.
4. Validate the catalog:

```bash
rastion validate-cards
```

5. Run the official TSP suites:

```bash
rastion eval-suite tsplib-small-v1
rastion eval-suite tsplib-medium-v1
rastion eval-suite tsplib-large-v1
```

6. Regenerate site artifacts:

```bash
rastion build-site-data
```

For a repo-verified walkthrough, use [docs/add-a-tsp-solver.md](docs/add-a-tsp-solver.md).

## Listing Rules

- Phase 1 public listings are `tsp` only.
- Public listings require executable adapters plus solver cards.
- Official leaderboards are suite-scoped and curated.
- The current official suites are a bootstrap track, not a claim of broad TSP supremacy.
- Work beyond TSP is future scope, not current product surface.

## Engineering Rules

- Keep solver cards honest about limits, optional dependencies, and failure modes.
- Do not claim support beyond the TSP surface the adapter actually runs today.
- Keep the TSP arena TSP-specific until multi-route semantics are added.
- Preserve non-breaking CLI behavior when extending the publication flow.
