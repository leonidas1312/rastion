# Use Rastion Yourself

This is the shortest repo-verified path from a fresh clone to a local build you can inspect.

## Recommended path

```bash
./scripts/release_smoke.sh
```

That script installs the full local verification dependencies, runs the Python test suite, validates the catalog, regenerates the public TSP artifacts, and builds the Astro site.

## Manual path

Create a local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ortools]"
```

Run the Python checks:

```bash
python -m pytest -q -s
python -m rastion validate-cards
```

Generate the public artifacts:

```bash
python -m rastion build-site-data --iters 800 --seed 0 --emit-every 50
```

Build the website:

```bash
cd web
npm ci
npm run build
```

## What to inspect after the build

- `web/public/data/catalog.json`
- `web/public/data/leaderboards.json`
- `web/public/data/evals/*.json`
- `web/public/data/tsp_arena.json`
- `web/dist/index.html`

The public release should read consistently as a TSP-only product with three visible surfaces: solver cards, reproducible evals, and replayable TSPLIB demos.
