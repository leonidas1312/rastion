# Rastion Phase 1 Launch Draft

Rastion is now a routing-first solver hub for Python.

What is live today:

- executable solver cards
- official TSP bootstrap evals
- replayable TSPLIB arena artifacts
- contributor workflow for adding TSP solvers

Why the wedge is narrow:

- routing is more coherent than "all optimization"
- TSP is already executable and visual in the repo
- the public contracts are designed so CVRP can follow without a product rewrite

What we are not claiming:

- VRP support today
- multi-route visualization today
- a universal "best routing solver" ranking

Why the current evals are called bootstrap suites:

- there are three official suites
- each suite currently contains one TSPLIB instance from the repo corpus
- seeds and budgets are fixed
- artifacts are published alongside the website

Recommended one-paragraph launch copy:

> Rastion is a GitHub-native routing solver hub for Python: executable solver cards, official TSP bootstrap evals, and replayable TSPLIB demos. Phase 1 is intentionally narrow and honest. TSP is fully supported today, VRP is next, and every published solver entry includes an adapter, metadata, and reproducible artifacts instead of marketing claims.

Suggested short social post:

> Shipping Rastion Phase 1: a routing-first solver hub for Python. Live now: executable TSP solver cards, bootstrap TSPLIB leaderboards, and replayable route demos. Narrow on purpose. Honest about limits. Built so CVRP can land next without redesign.
