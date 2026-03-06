# 2-Opt Local Search

2-opt is the first "real" optimization baseline in Rastion's routing hub. It starts from a greedy TSP route and repeatedly tries segment reversals to reduce total distance.

## Why it exists

- gives the hub a stronger official baseline than pure greedy construction
- demonstrates streaming iterative improvement
- acts as the reference shape for future routing heuristics

## Current limits

- single-route TSP only
- no explicit warm-start contract yet
- budget-sensitive on medium and large TSPLIB instances
