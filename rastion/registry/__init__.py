"""Registry public API."""

from rastion.registry.loader import AutoSolver, Problem, Solver
from rastion.registry.manager import (
    add_problem,
    export_problem,
    init_registry,
    install_problem,
    install_solver_from_url,
    list_problems,
    list_solvers,
    remove_problem,
)

__all__ = [
    "AutoSolver",
    "Problem",
    "Solver",
    "add_problem",
    "export_problem",
    "init_registry",
    "install_problem",
    "install_solver_from_url",
    "list_problems",
    "list_solvers",
    "remove_problem",
]
