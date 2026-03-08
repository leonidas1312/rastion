# Rastion

Rastion is the evaluation and publishing layer for executable routing solvers.

The current public release is intentionally narrow:

- TSP-first
- suite-scoped result exports
- replayable TSPLIB arena bundles
- no global ranking across unrelated problem families

## What It Gives You

- published solver cards under `catalog/solvers/*`
- versioned suite definitions under `catalog/suites/*.json`
- generated public exports under `web/public/data/*`
- a replay surface for exported TSPLIB runs
- one repo-verified flow from local adapter to public evidence

## 5-Minute Quickstart

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the full verification stack:

```bash
pip install -e ".[dev,ortools]"
```

Run the repo-verified flow:

```bash
./scripts/release_smoke.sh
```

That script installs dependencies, runs tests, validates cards, regenerates public data, and builds the Astro site.

If you want to inspect the site locally afterward:

```bash
cd web
npm install
npm run dev
```

## Public Surfaces

### Solver cards

Each public card points to a real adapter and keeps the install path, method, references, limits, and failure modes visible.

### Arena

Arena is the replay surface. It shows exported route geometry, progress traces, and gap-to-BKS information for published TSPLIB runs.

### Evaluation contract

Public claims come from generated artifacts, not handwritten marketing copy. The current public contract is:

- `web/public/data/catalog.json`
- `web/public/data/suites.json`
- `web/public/data/leaderboards.json`
- `web/public/data/evals/*.json`
- `web/public/data/tsp_arena.json`

`leaderboards.json` remains a compatibility filename. Public copy should read these as result exports, not as a promise of universal leaderboard semantics.

## Published Integrations

Rastion does not need to become a generic wrapper framework to work with external tools.

The guardrail is simple: an integration only becomes part of the public surface once it has:

1. a working adapter
2. a public solver card
3. honest documentation
4. generated suite output
5. optional Arena participation when the adapter is TSP-capable

Today the public catalog includes transparent built-in baselines plus one OR-Tools-backed comparison point.

## Run The Publication Flow Manually

Validate solver cards and suite specs:

```bash
rastion validate-cards
```

Generate a suite export:

```bash
rastion eval-suite tsplib-small-v1
```

Rebuild the public website data:

```bash
rastion build-site-data --iters 800 --seed 0 --emit-every 50
```

Build the site:

```bash
cd web
npm ci
npm run build
```

## Contributing

Public listings require:

1. a working adapter under `plugins_local/`
2. a valid solver card under `catalog/solvers/<solver-id>/card.json`
3. a solver detail markdown file
4. successful validation with `rastion validate-cards`
5. generated suite output from the current TSP suite set
6. regenerated public site artifacts

Start with:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/add-a-tsp-solver.md](docs/add-a-tsp-solver.md)
- [docs/use-rastion-yourself.md](docs/use-rastion-yourself.md)

## Non-Goals

`v0.1` does not claim:

- CVRP or VRPTW execution
- multi-vehicle route rendering
- ecosystem breadth without published evidence
- global rankings across problem families

Future expansion only makes sense after the current publication contract gets enough usage and scrutiny.

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

- repository pages default to `/<repo>/`
- local dev defaults to `/`
- override with `SITE_BASE=/rastion/` if needed

## Tests

```bash
python3 -m pytest -q -s
```
