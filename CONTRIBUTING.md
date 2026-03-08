# Contributing

Rastion `v0.1` is a TSP-first publishing surface for executable routing solvers. Contributions should preserve that contract: explicit adapters, honest metadata, generated evidence, and restrained public claims.

## Add A Solver

1. Add an adapter under `plugins_local/`.
2. Add a solver card at `catalog/solvers/<solver-id>/card.json`.
3. Add `catalog/solvers/<solver-id>/README.md`.
4. Validate the catalog:

```bash
rastion validate-cards
```

5. Run the current TSP suites:

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

## Publication Rules

- Phase 1 public listings are `tsp` only.
- Public listings require executable adapters plus solver cards.
- Public result exports are suite-scoped and curated.
- The current suites are a bootstrap track, not a claim of broad TSP supremacy.
- External frameworks belong on the public site only after they ship as real adapters with generated evidence.

## Engineering Rules

- Keep solver cards honest about limits, optional dependencies, and failure modes.
- Do not claim support beyond the TSP surface the adapter actually runs today.
- Keep the TSP arena TSP-specific until multi-route semantics are added.
- Preserve non-breaking CLI behavior when extending the publication flow.
