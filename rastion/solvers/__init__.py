"""Solver package."""

from .registry import (
    RegisteredSolver,
    clear_registry,
    get_registered_solver,
    get_solver,
    list_solvers,
    register_solver,
)

__all__ = [
    "RegisteredSolver",
    "clear_registry",
    "register_solver",
    "get_solver",
    "get_registered_solver",
    "list_solvers",
]
