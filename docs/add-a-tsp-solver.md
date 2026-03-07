# Add a TSP Solver to Rastion

This walkthrough matches the Phase 1 repository layout as shipped during the content freeze.

## Goal

Produce a public TSP listing with:

- a discoverable adapter
- a validated solver card
- a solver detail README that renders on the site
- official bootstrap suite results
- regenerated static site artifacts

## Prerequisites

Install the package in editable mode:

```bash
pip install -e .
```

If your adapter depends on OR-Tools:

```bash
pip install -e '.[ortools]'
```

All repo docs use the installed console entrypoint:

```bash
rastion --help
```

## 1. Add the adapter

Create a package under `plugins_local/` that exposes `get_solver()`.

Reference implementations already in the repo:

- `plugins_local/rastion_tsp_nearest`
- `plugins_local/rastion_tsp_two_opt`
- `plugins_local/rastion_tsp_ortools`

Phase 1 requirements:

- supports `tsp`
- runs locally from the repo
- behaves honestly with respect to dependencies and limits

## 2. Add the solver card

Create:

- `catalog/solvers/<solver-id>/card.json`
- `catalog/solvers/<solver-id>/README.md`

The card provides structured metadata.
The README is the narrative layer that the site renders on the solver detail page.

Minimum content to include in the README:

- what the solver is
- when to use it
- what it cannot do
- known failure modes
- interpretation of current suite results

## 3. Validate the catalog contract

```bash
rastion validate-cards
```

Validation checks:

- duplicate ids
- missing suite specs
- missing solver detail markdown
- missing adapter paths
- undiscoverable runtime solver names
- non-TSP Phase 1 cards or suites

## 4. Run the official bootstrap suites

```bash
rastion eval-suite tsplib-small-v1
rastion eval-suite tsplib-medium-v1
rastion eval-suite tsplib-large-v1
```

Why "bootstrap":

- each official Phase 1 suite currently contains one TSPLIB instance
- this keeps the evaluation reproducible and honest while the corpus is still small
- do not market the resulting ranks as a universal TSP verdict

## 5. Regenerate the website artifacts

```bash
rastion build-site-data
```

This writes:

- `web/public/data/catalog.json`
- `web/public/data/suites.json`
- `web/public/data/leaderboards.json`
- `web/public/data/evals/*.json`
- `web/public/data/tsp_arena.json`

## 6. Check the website locally

```bash
cd web
npm install
npm run dev
```

Verify:

- your solver appears in the catalog
- its README renders on the detail page
- suite coverage is visible
- failure modes and references are present

## 7. Prepare the PR

Include:

- adapter package
- solver card
- solver README
- regenerated JSON artifacts if the repo expects them
- a truthful description of method class, dependency requirements, and limits

Avoid:

- claiming VRP support
- claiming broad routing superiority from the bootstrap suites
- hiding dependency failures or unsupported constraints
