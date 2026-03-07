#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -e "$ROOT_DIR[dev,ortools]"
"$VENV_DIR/bin/python" -m pytest -q -s
"$VENV_DIR/bin/python" -m rastion validate-cards
"$VENV_DIR/bin/python" -m rastion build-site-data --iters 800 --seed 0 --emit-every 50

cd "$ROOT_DIR/web"
npm ci
npm run build
