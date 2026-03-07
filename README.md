# Rastion

Rastion is a routing solver hub for Python: executable solver cards, official TSP evals, and replayable TSPLIB demos.

The current wedge is deliberate:

- `routing` is the product story
- `TSP` is the fully supported vertical today
- `VRP` is roadmap work, not implied current capability

## Live Demo

GitHub Pages publishes the static hub from generated JSON artifacts:

- solver catalog
- official TSP leaderboards
- TSPLIB arena
- suite definitions

## Why Routing First?

Routing is narrow enough to be scientifically coherent and broad enough to grow into a real ecosystem. Rastion starts
with TSP because the repo already has executable TSPLIB support, route visualization, and baseline heuristics. The
public schemas are VRP-ready so CVRP and richer routing variants can land without redesigning the catalog contract.

## TSP Today, VRP Next

Phase 1 supports:

- executable TSP solver adapters
- solver cards in `catalog/solvers/*/card.json`
- official TSPLIB suites in `catalog/suites/*.json`
- suite-scoped leaderboards
- replayable TSP arena artifacts

Phase 1 does not support:

- CVRP or VRPTW execution
- multi-vehicle route rendering
- global cross-family routing rankings

## 5-Minute Quickstart

Install the package:

```bash
pip install -e .
```

With the optional OR-Tools adapter enabled:

```bash
pip install -e '.[ortools]'
```

Generate all static site artifacts:

```bash
rastion build-site-data
```

The installed `rastion` console script is the recommended entrypoint for docs and contributor workflows.
`python -m rastion ...` remains compatible if you prefer module execution.

Run the Astro site:

```bash
cd web
npm install
npm run dev
```

## Core Commands

Validate solver cards and suite specs:

```bash
rastion validate-cards
```

Export the public catalog and suite metadata:

```bash
rastion export-catalog --out web/public/data/catalog.json
rastion export-suites --out web/public/data/suites.json
```

Run an official suite and export the suite artifact:

```bash
rastion eval-suite tsplib-small-v1
```

Export all leaderboards:

```bash
rastion export-leaderboards --out web/public/data/leaderboards.json
```

Generate the TSP arena artifact:

```bash
rastion tsp-arena --out web/public/data/tsp_arena.json --iters 2000 --seed 0 --emit-every 50
```

Compatibility alias:

```bash
rastion demo
rastion demo-site
```

## Add A TSP Solver

Phase 1 public listings require:

1. a working adapter under `plugins_local/`
2. a valid solver card under `catalog/solvers/<solver-id>/card.json`
3. a solver detail markdown file
4. successful validation with `rastion validate-cards`
5. official suite output from the TSP suite set

See [CONTRIBUTING.md](CONTRIBUTING.md) for the exact flow and [docs/add-a-tsp-solver.md](docs/add-a-tsp-solver.md)
for the full walkthrough.

## Official Suite Policy

Leaderboards are:

- `TSP` only in Phase 1
- versioned
- curated
- suite-scoped
- generated from fixed seeds and budgets

Rastion intentionally does not publish a fake global "best routing solver" ranking.

Current suite scope is intentionally labeled as a bootstrap track:

- there are three official Phase 1 suites
- each suite currently contains one TSPLIB instance
- this makes the methodology honest, reproducible, and easy to audit
- it does not justify broad claims about general TSP dominance

Read the public framing in:

- [docs/launch-announcement.md](docs/launch-announcement.md)
- [docs/add-a-tsp-solver.md](docs/add-a-tsp-solver.md)

## Repo Layout

- `catalog/solvers/`: solver cards and detail markdown
- `catalog/suites/`: official suite definitions
- `docs/`: launch and contributor content for the freeze window
- `plugins_local/`: executable solver adapters
- `rastion/catalog/`: catalog loading, validation, exports, and evals
- `rastion/tsp/`: TSPLIB ingestion and TSP arena utilities
- `web/`: Astro site

## Roadmap

Phase 2 targets basic CVRP support:

- `CVRPProblem`
- multi-route result schema
- CVRP suite definitions
- multi-vehicle visualizations

Phase 3 can expand into broader routing variants once the contracts are stable.

## GitHub Pages

`web/astro.config.mjs` derives the base path from CI automatically.

Project Pages:

- repository pages default to `/<repo>/`
- local dev defaults to `/`
- override with `SITE_BASE=/rastion/` if needed

## Tests

```bash
pytest -q
```
