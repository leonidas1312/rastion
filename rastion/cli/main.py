"""Alternate CLI module path for `python -m rastion.cli.main`."""

from __future__ import annotations

import sys

from rastion.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
