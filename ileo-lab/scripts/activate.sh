#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Run this script with: source scripts/activate.sh"
  exit 1
fi

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$LAB_ROOT/.." && pwd)"
VENV_PATH="$LAB_ROOT/.venv"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Missing virtual environment at $VENV_PATH"
  echo "Run: ./scripts/bootstrap.sh"
  return 1
fi

# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
export RASTION_HOME="$LAB_ROOT/.rastion-home"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
else
  export PYTHONPATH="$REPO_ROOT"
fi

echo "ileo-lab activated"
echo "RASTION_HOME=$RASTION_HOME"
