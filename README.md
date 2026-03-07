# Rastion

Rastion publishes executable TSP solver cards, reproducible suite results, and replayable TSPLIB arena exports.

`v0.1` is intentionally narrow:

- TSP only
- suite-scoped results
- small, auditable bootstrap suites
- no global ranking across unrelated problem families

## What You Can Do Today

- browse solver cards under `catalog/solvers/*`
- run official TSPLIB suites from `catalog/suites/*.json`
- publish suite-scoped leaderboards
- replay exported TSPLIB routes in the arena
- use the same generated artifacts locally and on GitHub Pages

## 5-Minute Setup

Create a local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Minimal install:

```bash
pip install -e .
```

Full local verification setup:

```bash
pip install -e ".[dev,ortools]"
```

Run the published local verification flow:

```bash
./scripts/release_smoke.sh
```

That script creates and reuses `.venv/` automatically.

Start the site locally:

```bash
cd web
npm install
npm run dev
```

## Run The Official Flow

Validate solver cards and suite specs:

```bash
rastion validate-cards
```

Run an official suite and export its artifact:

```bash
rastion eval-suite tsplib-small-v1
```

Generate the public website artifacts:

```bash
rastion build-site-data --iters 800 --seed 0 --emit-every 50
```

Build the Astro site:

```bash
cd web
npm ci
npm run build
```

The public website contract for `v0.1` is:

- `web/public/data/catalog.json`
- `web/public/data/suites.json`
- `web/public/data/leaderboards.json`
- `web/public/data/evals/*.json`
- `web/public/data/tsp_arena.json`

## Use Rastion Yourself

The shortest repo-verified path from clone to visible output is documented in:

- [docs/use-rastion-yourself.md](docs/use-rastion-yourself.md)
- [docs/publish-checklist.md](docs/publish-checklist.md)

## Current Solver Track

The current public track is intentionally small:

- `Nearest Neighbor` is the transparent baseline
- `2-Opt Local Search` is the stronger built-in baseline
- `OR-Tools Routing` is the optional external comparison point

The current official suites are also intentionally small:

- `tsplib-small-v1`
- `tsplib-medium-v1`
- `tsplib-large-v1`

Each suite currently contains one TSPLIB instance. That is enough to make the methodology honest, reproducible, and easy to audit. It is not enough to claim broad TSP dominance.

## Limits And Non-Goals

`v0.1` does not support:

- CVRP or VRPTW execution
- multi-vehicle route rendering
- global rankings across problem families

Future work can expand beyond TSP after the current contracts have enough usage and test coverage to justify it.

## Why Publish Now

Rastion is at the right stage to publish because the core loop is real:

- cards are executable, not just marketing copy
- results are reproducible and artifact-backed
- route demos are replayable from exported TSPLIB runs

Publishing now is meant to answer one question: is this useful enough to keep investing in?

## Contributing A Solver

Public TSP listings require:

1. a working adapter under `plugins_local/`
2. a valid solver card under `catalog/solvers/<solver-id>/card.json`
3. a solver detail markdown file
4. successful validation with `rastion validate-cards`
5. official suite output from the TSP suite set
6. regenerated public site artifacts

Use:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/add-a-tsp-solver.md](docs/add-a-tsp-solver.md)

## Repo Layout

- `catalog/solvers/`: public solver cards and detail markdown
- `catalog/suites/`: official suite definitions
- `docs/`: local usage, publish, and contributor guides
- `plugins_local/`: executable solver adapters
- `rastion/catalog/`: validation, exports, and suite evals
- `rastion/tsp/`: TSPLIB ingestion and arena utilities
- `web/`: Astro site

## GitHub Pages

`web/astro.config.mjs` derives the base path from CI automatically.

Project Pages:

- repository pages default to `/<repo>/`
- local dev defaults to `/`
- override with `SITE_BASE=/rastion/` if needed

## Tests

```bash
python3 -m pytest -q -s
```
