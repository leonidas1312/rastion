"""Rastion: portable optimization problems and solver plugins."""

from rastion.benchmark import compare, run_benchmark_suite
from rastion.registry import AutoSolver, Problem, Solver
from rastion.version import __version__

__all__ = [
    "__version__",
    "AutoSolver",
    "Problem",
    "Solver",
    "compare",
    "run_benchmark_suite",
]
