from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rastion.solvers.schema import SolverMetadata
from rastion.solvers.base import Solver


SolverFactory = Callable[[], Solver]
SolverFactoryOrInstance = Solver | SolverFactory


@dataclass(slots=True)
class RegisteredSolver:
    name: str
    solver_factory_or_instance: SolverFactoryOrInstance
    metadata: SolverMetadata
    _solver_instance: Solver | None = None

    @property
    def solver(self) -> Solver:
        if self._solver_instance is not None:
            return self._solver_instance

        candidate = self.solver_factory_or_instance
        if isinstance(candidate, Solver):
            self._solver_instance = candidate
            return candidate

        instance = candidate()
        if not isinstance(instance, Solver):
            raise TypeError(f"Factory for '{self.name}' returned unsupported type: {type(instance)!r}")
        self._solver_instance = instance
        return instance


_REGISTRY: dict[str, RegisteredSolver] = {}


def clear_registry() -> None:
    _REGISTRY.clear()


def register_solver(name: str, solver_factory_or_instance: SolverFactoryOrInstance, metadata: SolverMetadata) -> None:
    if not name:
        raise ValueError("Solver name must be non-empty")

    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("Solver name must be non-empty")

    row = RegisteredSolver(
        name=normalized_name,
        solver_factory_or_instance=solver_factory_or_instance,
        metadata=metadata.model_copy(update={"name": normalized_name}),
    )

    solver = row.solver
    if solver.name != normalized_name:
        raise ValueError(
            f"Registry name '{normalized_name}' does not match solver.name '{solver.name}'"
        )

    _REGISTRY[normalized_name] = row


def get_registered_solver(name: str) -> RegisteredSolver:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Solver '{name}' is not registered") from exc


def get_solver(name: str) -> Solver:
    return get_registered_solver(name).solver


def list_solvers() -> list[RegisteredSolver]:
    return list(_REGISTRY.values())
