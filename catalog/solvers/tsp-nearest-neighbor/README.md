# Nearest Neighbor

Nearest Neighbor is Rastion's simplest TSP reference solver. It is intended to be readable, deterministic for a fixed seed, and cheap enough to run in every official suite.

## Why it exists

- establishes a transparent routing baseline
- emits progress events for the arena
- gives contributors a minimal example of a valid solver adapter

## Current limits

- single-route TSP only
- no local search refinement
- not competitive on larger TSPLIB instances
