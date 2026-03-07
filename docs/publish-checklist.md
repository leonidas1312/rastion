# Publish Checklist

Use this checklist before pushing the public `v0.1` release.

## Product framing

- README starts with the public TSP hub catchphrase.
- Homepage, About, Solvers, Leaderboards, and Arena all read as one coherent TSP product.
- No public-facing page leads with broader routing or VRP framing.
- Old pages that do not fit `v0.1` are redirected or noindexed.

## Repo verification

- `./scripts/release_smoke.sh`
- `python -m pytest -q -s`
- `python -m rastion validate-cards`
- `python -m rastion build-site-data --iters 800 --seed 0 --emit-every 50`
- `cd web && npm ci && npm run build`

## Public artifact check

- `catalog.json` contains the expected solver cards.
- `suites.json` contains the three bootstrap-track TSPLIB suites.
- `leaderboards.json` is suite-scoped and artifact-backed.
- `evals/*.json` exist for each official suite.
- `tsp_arena.json` loads the TSPLIB arena instances.

## Documentation check

- README links only point to files that exist.
- `docs/use-rastion-yourself.md` is accurate.
- `docs/add-a-tsp-solver.md` matches the real flow.
- `CONTRIBUTING.md` and the website contributor page say the same thing.

## Release decision

If the checklist passes, publish the repo and use the current release to test whether people care enough to justify the next phase.
