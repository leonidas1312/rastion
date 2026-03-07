# OR-Tools Routing

This adapter exposes OR-Tools as an experimental TSP listing inside Rastion's public hub.

## Why it exists

- provides a familiar external optimization backend
- shows how optional dependencies surface in solver cards and eval runs
- gives contributors an example of a non-core adapter with a stronger search backend

## Current limits

- requires the `ortools` extra
- still represented as TSP only in the v0.1 catalog
- broader routing capabilities are intentionally outside the current public release
