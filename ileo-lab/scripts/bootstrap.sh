#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$LAB_ROOT/.." && pwd)"
VENV_PATH="$LAB_ROOT/.venv"

python3 -m venv "$VENV_PATH"
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e "$REPO_ROOT[dev,tui]"

echo "Bootstrap complete."
echo "Use: source $LAB_ROOT/scripts/activate.sh"
