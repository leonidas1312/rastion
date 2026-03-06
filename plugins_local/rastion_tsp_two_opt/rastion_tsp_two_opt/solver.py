from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from rastion.solvers.base import ProgressEvent, Solution, Solver
from rastion.tsp.types import TSPProblem
from rastion.tsp.utils import cycle_length, nearest_neighbor_route


def _two_opt_swap(route: list[int], i: int, k: int) -> list[int]:
    return route[:i] + list(reversed(route[i : k + 1])) + route[k + 1 :]


@dataclass
class TSPTwoOptSolver(Solver):
    name: str = "tsp_two_opt"
    supports: list[str] = None
    max_size: int = 2_000
    hardware: list[str] = None
    quality: float = 0.78

    def __post_init__(self) -> None:
        if self.supports is None:
            self.supports = ["tsp"]
        if self.hardware is None:
            self.hardware = ["cpu"]

    def _run(self, problem_ir: TSPProblem, **kwargs: Any) -> tuple[Solution, list[ProgressEvent]]:
        if problem_ir.problem_type != "tsp":
            raise ValueError(f"{self.name} expects tsp problems")

        seed = int(kwargs.get("seed", 0))
        iters = int(kwargs.get("iters", 4_000))
        emit_every = max(int(kwargs.get("emit_every", 50)), 1)
        time_budget_ms = kwargs.get("time_budget_ms")
        budget = float(time_budget_ms) if time_budget_ms is not None else None

        n = problem_ir.n_vars
        started = time.perf_counter()
        rng = np.random.default_rng(seed)

        route = nearest_neighbor_route(problem_ir.distance_matrix, problem_ir.depot, seed=seed)
        best_route = route[:]
        best_value = cycle_length(problem_ir.distance_matrix, best_route)

        events: list[ProgressEvent] = [ProgressEvent(t_ms=0, iter=0, best_value=best_value)]

        for step in range(1, iters + 1):
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if budget is not None and elapsed_ms >= budget:
                break

            i = int(rng.integers(1, n - 1))
            k = int(rng.integers(i + 1, n))
            candidate = _two_opt_swap(best_route, i, k)
            candidate_value = cycle_length(problem_ir.distance_matrix, candidate)

            if candidate_value < best_value:
                best_value = candidate_value
                best_route = candidate

            if step % emit_every == 0 or step == iters:
                events.append(
                    ProgressEvent(
                        t_ms=max(int((time.perf_counter() - started) * 1000.0), 0),
                        iter=step,
                        best_value=best_value,
                    )
                )

        runtime_ms = max(int((time.perf_counter() - started) * 1000.0), 0)
        if events[-1].iter != iters:
            events.append(ProgressEvent(t_ms=runtime_ms, iter=iters, best_value=best_value))

        solution = Solution(
            solver_name=self.name,
            best_x=np.asarray(best_route, dtype=int),
            best_value=best_value,
            metadata={
                "seed": seed,
                "iters": iters,
                "route": best_route,
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
    return TSPTwoOptSolver()
