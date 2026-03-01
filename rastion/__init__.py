"""Rastion: portable decision plugins and solver plugins."""

from rastion.benchmark import compare, run_benchmark_suite
from rastion.registry import AutoSolver, DecisionPlugin, Problem, Solver
from rastion.version import __version__

__all__ = [
    "__version__",
    "AutoSolver",
    "DecisionPlugin",
    "Problem",
    "Solver",
    "compare",
    "run_benchmark_suite",
]
