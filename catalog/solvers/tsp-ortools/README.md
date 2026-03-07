# OR-Tools Routing

This adapter exposes Google OR-Tools as an experimental TSP solver inside Rastion.

## Why it is included

- Provides an external comparison point alongside the built-in heuristics.
- Exercises the optional-dependency path used by the local eval and export flow.
- Shows how a third-party routing backend can be wrapped in the same solver card and suite interfaces.

## Current limits

- Requires the `ortools` extra.
- The current adapter covers single-route TSP only.
- Broader vehicle-routing features are outside the v0.1 public surface.
