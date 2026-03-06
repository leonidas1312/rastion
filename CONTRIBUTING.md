# Contributing

Rastion Phase 1 is a routing-first hub with real TSP support. Contributions should preserve that contract: executable
adapters, explicit metadata, and reproducible suite outputs.

## Add A Solver

1. Add an adapter under `plugins_local/`.
2. Add a solver card at `catalog/solvers/<solver-id>/card.json`.
3. Add `catalog/solvers/<solver-id>/README.md`.
4. Validate the catalog:

```bash
python -m rastion validate-cards
```

5. Run the official TSP suites:

```bash
python -m rastion eval-suite tsplib-small-v1
python -m rastion eval-suite tsplib-medium-v1
python -m rastion eval-suite tsplib-large-v1
```

6. Regenerate site artifacts:

```bash
python -m rastion build-site-data
```

## Listing Rules

- Phase 1 public listings are `tsp` only.
- `routing` is the umbrella taxonomy.
- `vrp` and richer routing variants are roadmap work.
- Public listings require executable adapters plus solver cards.
- Official leaderboards are suite-scoped and curated.

## Engineering Rules

- Keep solver cards honest about limits, optional dependencies, and failure modes.
- Do not claim support for VRP variants that the adapter cannot actually run.
- Keep the TSP arena TSP-specific until multi-route semantics are added.
- Preserve non-breaking CLI behavior when extending the publication flow.
