from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from rastion.solvers.base import ProgressEvent, Solution, Solver
from rastion.tsp.types import TSPProblem
from rastion.tsp.utils import cycle_length

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
except ImportError as exc:  # pragma: no cover - exercised where ortools is installed
    pywrapcp = None
    routing_enums_pb2 = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class ORToolsTSPSolver(Solver):
    name: str = "tsp_ortools"
    supports: list[str] = None
    max_size: int = 5_000
    hardware: list[str] = None
    quality: float = 0.95

    def __post_init__(self) -> None:
        if self.supports is None:
            self.supports = ["tsp"]
        if self.hardware is None:
            self.hardware = ["cpu"]

    def _ensure_available(self) -> None:
        if pywrapcp is None or routing_enums_pb2 is None:
            message = "ortools is not installed. Install with: pip install -e '.[ortools]'"
            if _IMPORT_ERROR is not None:
                message += f" ({_IMPORT_ERROR})"
            raise RuntimeError(message)

    def solve(self, problem_ir: TSPProblem, **kwargs: Any) -> Solution:
        self._ensure_available()
        if problem_ir.problem_type != "tsp":
            raise ValueError(f"{self.name} expects tsp problems")

        started = time.perf_counter()
        seed = int(kwargs.get("seed", 0))
        time_budget_ms = int(kwargs.get("time_budget_ms", 2_000))

        n = problem_ir.n_vars
        depot = int(problem_ir.depot)
        matrix = np.asarray(problem_ir.distance_matrix, dtype=float)
        int_matrix = np.rint(matrix).astype(int)

        manager = pywrapcp.RoutingIndexManager(n, 1, depot)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(int_matrix[from_node, to_node])

        transit_callback = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback)

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.FromMilliseconds(max(time_budget_ms, 1))

        assignment = routing.SolveWithParameters(params)
        if assignment is None:
            raise RuntimeError("OR-Tools failed to produce a route")

        route = [depot]
        index = routing.Start(0)
        while not routing.IsEnd(index):
            index = assignment.Value(routing.NextVar(index))
            node = manager.IndexToNode(index)
            route.append(int(node))

        if route[-1] != depot:
            route.append(depot)

        best_value = cycle_length(matrix, route)
        runtime_ms = max(int((time.perf_counter() - started) * 1000.0), 0)

        return Solution(
            solver_name=self.name,
            best_x=np.asarray(route, dtype=int),
            best_value=best_value,
            metadata={
                "seed": seed,
                "time_budget_ms": time_budget_ms,
                "route": route,
                "runtime_ms": runtime_ms,
            },
        )

    def solve_trace(self, problem_ir: TSPProblem, **kwargs: Any) -> tuple[Solution, list[ProgressEvent]]:
        started = time.perf_counter()
        solution = self.solve(problem_ir, **kwargs)
        runtime_ms = max(int(solution.metadata.get("runtime_ms", (time.perf_counter() - started) * 1000.0)), 0)
        event = ProgressEvent(
            t_ms=runtime_ms,
            iter=max(int(kwargs.get("iters", 1)), 1),
            best_value=solution.best_value,
        )
        return solution, [event]

    def solve_stream(self, problem_ir: TSPProblem, **kwargs: Any) -> Iterator[ProgressEvent]:
        _, events = self.solve_trace(problem_ir, **kwargs)
        yield from events


def get_solver() -> Solver:
    return ORToolsTSPSolver()
