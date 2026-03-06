# OR-Tools Routing

This adapter exposes OR-Tools as an experimental Phase 1 TSP listing inside Rastion's routing hub.

## Why it exists

- provides a familiar external routing backend
- shows how optional dependencies surface in solver cards and eval runs
- gives contributors an example of a non-core adapter with a stronger search backend

## Current limits

- requires the `ortools` extra
- still represented as TSP only in the Phase 1 catalog
- richer VRP capabilities are intentionally deferred until the routing interfaces expand
