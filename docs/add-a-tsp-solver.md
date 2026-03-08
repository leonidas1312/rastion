# Add A TSP Solver

This is the repo-verified path for turning a local adapter into a public Rastion listing.

## 1. Install the full local toolchain

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ortools]"
```

## 2. Add the adapter under `plugins_local/`

Your plugin must expose `get_solver()` and register a solver that supports `tsp`.

Existing examples:

- `plugins_local/rastion_tsp_nearest`
- `plugins_local/rastion_tsp_two_opt`
- `plugins_local/rastion_tsp_ortools`

## 3. Add the public card and detail page

Create:

- `catalog/solvers/<solver-id>/card.json`
- `catalog/solvers/<solver-id>/README.md`

Keep the card honest about:

- installation requirements
- limits
- determinism
- optional dependencies
- known failure modes
- whether the adapter wraps an external framework

## 4. Validate the catalog

```bash
python -m rastion validate-cards
```

Validation checks:

- card ids
- suite ids
- adapter path existence
- discoverability of the runtime solver
- presence of the detail markdown file

## 5. Run the official suites

```bash
python -m rastion eval-suite tsplib-small-v1
python -m rastion eval-suite tsplib-medium-v1
python -m rastion eval-suite tsplib-large-v1
```

Public evidence in `v0.1` is suite-scoped. Do not treat one suite run as proof of broad TSP superiority.

## 6. Regenerate the public artifacts

```bash
python -m rastion build-site-data --iters 800 --seed 0 --emit-every 50
```

Then build the site:

```bash
cd web
npm ci
npm run build
```

## 7. Check the public surfaces

Verify that the new solver appears in:

- the solver catalog
- its detail page
- suite-scoped result exports
- the TSPLIB arena export if it participates there

## 8. Run the release smoke flow

Before publishing or opening a PR, run:

```bash
./scripts/release_smoke.sh
```

That is the fastest way to ensure the repo still behaves like a coherent public publication surface.
