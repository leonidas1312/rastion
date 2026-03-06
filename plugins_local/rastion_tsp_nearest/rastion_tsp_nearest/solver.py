from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from rastion.solvers.base import ProgressEvent, Solution, Solver
from rastion.tsp.types import TSPProblem
from rastion.tsp.utils import cycle_length, nearest_neighbor_route


@dataclass
class TSPNearestNeighborSolver(Solver):
    name: str = "tsp_nearest_neighbor"
    supports: list[str] = None
    max_size: int = 2_000
    hardware: list[str] = None
    quality: float = 0.62

    def __post_init__(self) -> None:
        if self.supports is None:
            self.supports = ["tsp"]
        if self.hardware is None:
            self.hardware = ["cpu"]

    def _run(self, problem_ir: TSPProblem, **kwargs: Any) -> tuple[Solution, list[ProgressEvent]]:
        if problem_ir.problem_type != "tsp":
            raise ValueError(f"{self.name} expects tsp problems")

        seed = int(kwargs.get("seed", 0))
        emit_every = max(int(kwargs.get("emit_every", 1)), 1)
        started = time.perf_counter()

        route = nearest_neighbor_route(problem_ir.distance_matrix, problem_ir.depot, seed=seed)
        events: list[ProgressEvent] = []

        prefix = [route[0]]
        for idx, node in enumerate(route[1:], start=1):
            prefix.append(node)
            if idx < len(route) - 1:
                partial_len = cycle_length(problem_ir.distance_matrix, prefix + [problem_ir.depot])
            else:
                partial_len = cycle_length(problem_ir.distance_matrix, prefix)

            if idx % emit_every == 0 or idx == len(route) - 1:
                events.append(
                    ProgressEvent(
                        t_ms=max(int((time.perf_counter() - started) * 1000.0), 0),
                        iter=idx,
                        best_value=partial_len,
                    )
                )

        if not events:
            events = [ProgressEvent(t_ms=0, iter=0, best_value=cycle_length(problem_ir.distance_matrix, route))]

        best_value = cycle_length(problem_ir.distance_matrix, route)
        runtime_ms = max(int((time.perf_counter() - started) * 1000.0), 0)

        solution = Solution(
            solver_name=self.name,
            best_x=np.asarray(route, dtype=int),
            best_value=best_value,
            metadata={
                "seed": seed,
                "route": route,
                "runtime_ms": runtime_ms,
            },
        )
        return solution, events

    def solve(self, problem_ir: TSPProblem, **kwargs: Any) -> Solution:
        solution, _ = self._run(problem_ir, **kwargs)
        return solution

    def solve_stream(self, problem_ir: TSPProblem, **kwargs: Any) -> Iterator[ProgressEvent]:
        _, events = self._run(problem_ir, **kwargs)
        yield from events


def get_solver() -> Solver:
    return TSPNearestNeighborSolver()
