# ileo-lab

Fresh, isolated workspace for testing Rastion CLI, TUI, and core flows with custom heuristic packages.

## What This Lab Adds

- isolated runtime (`ileo-lab/.venv`, `ileo-lab/.rastion-home`)
- generated stress problem packages for:
  - binary knapsack (`lab_knapsack_stress`)
  - MaxCut-style QUBO (`lab_maxcut_stress`)
- production-style heuristic solver packages:
  - `grasp_binary_pro` (binary linear, GRASP + local search)
  - `qubo_tabu_pro` (QUBO, tabu search + restarts)
- scripted create/load/test workflow
- optional hub upload/load roundtrip script

## Quick Start

From repo root:

```bash
cd ileo-lab
./scripts/bootstrap.sh
source scripts/activate.sh
python scripts/lab.py all
```

## Commands

```bash
python scripts/lab.py create
python scripts/lab.py load
python scripts/lab.py test
python scripts/lab.py all
```

Optional hub upload/load smoke test:

```bash
python scripts/lab.py hub-roundtrip \
  --hub-url http://localhost:8000 \
  --github-token YOUR_GITHUB_TOKEN \
  --problem lab_knapsack_stress \
  --solver grasp_binary_pro
```

## TUI Manual Check

After `source scripts/activate.sh`:

```bash
rastion
```

The TUI will run against the isolated `RASTION_HOME=ileo-lab/.rastion-home` registry.

## Artifacts

Generated artifacts are written to `ileo-lab/workspace/`:

- `problems/` generated shareable problem folders
- `archives/` zipped solver packages for `rastion install-solver`
- `exports/` exported problem bundles
