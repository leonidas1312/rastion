# Rastion v0.1 Launch Draft

Rastion is now a public publishing surface for executable TSP solvers.

What is live today:

- executable solver cards
- reproducible TSP bootstrap evals
- replayable TSPLIB demos
- contributor workflow for adding TSP solvers

Why the release is narrow:

- TSP is already executable and visual in the repo
- the public cards, evals, and demos come from one auditable workflow
- the current scope is small enough to publish honestly

What we are not claiming:

- support beyond TSP today
- multi-route visualization today
- a universal "best solver" ranking

Why the current evals are called bootstrap suites:

- there are three official suites
- each suite currently contains one TSPLIB instance from the repo corpus
- seeds and budgets are fixed
- artifacts are published alongside the website

Recommended one-paragraph launch copy:

> Rastion is the evaluation and publishing layer for executable TSP solvers, with solver cards, reproducible evals, and replayable TSPLIB demos. The v0.1 release is intentionally narrow and honest: TSP only, suite-scoped result exports, and artifact-backed output instead of marketing claims.

Suggested short social post:

> Shipping Rastion v0.1: executable solver cards, reproducible TSPLIB result exports, and replayable demos. Narrow on purpose. Honest about limits. Publishing now to see if people actually want more.
