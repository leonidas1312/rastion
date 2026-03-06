from __future__ import annotations

from collections.abc import Callable

from rastion.solvers.base import Solver
from rastion.solvers.schema import SolverMetadata
from rastion.solvers.registry import (
    RegisteredSolver,
    clear_registry as _clear_registry,
    get_registered_solver,
    list_solvers,
    register_solver as _register_solver,
)


def clear_registry() -> None:
    _clear_registry()


def register_solver(
    solver: Solver | Callable[[], Solver],
    metadata: SolverMetadata,
) -> None:
    name = getattr(solver, "name", metadata.name)
    _register_solver(str(name), solver, metadata)


def get_solver(name: str) -> RegisteredSolver:
    return get_registered_solver(name)


def list_registered_solvers() -> list[RegisteredSolver]:
    return list_solvers()
