from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from rastion.ir.types import ProblemIR
from rastion.solvers.base import ProgressEvent, Solution, Solver, evaluate_qubo


@dataclass
class SimulatedAnnealingSolver(Solver):
    name: str = "simulated_annealing"
    supports: list[str] = None
    max_size: int = 200
    hardware: list[str] = None
    quality: float = 0.70

    def __post_init__(self) -> None:
        if self.supports is None:
            self.supports = ["qubo"]
        if self.hardware is None:
            self.hardware = ["cpu"]

    def _run(self, problem_ir: ProblemIR, **kwargs: Any) -> tuple[Solution, list[ProgressEvent]]:
        seed = int(kwargs.get("seed", 0))
        iters = int(kwargs.get("iters", 300))
        t_start = float(kwargs.get("t_start", 2.0))
        t_end = float(kwargs.get("t_end", 0.01))
        emit_every = max(int(kwargs.get("emit_every", 25)), 1)
        time_budget_ms = kwargs.get("time_budget_ms")
        budget = float(time_budget_ms) if time_budget_ms is not None else None

        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        n = problem_ir.n_vars

        x = rng.integers(0, 2, size=n, dtype=int)
        value = evaluate_qubo(problem_ir, x)
        best_x = x.copy()
        best_value = value

        events: list[ProgressEvent] = [ProgressEvent(t_ms=0, iter=0, best_value=best_value)]
        last_iter = 0

        for step in range(1, iters + 1):
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if budget is not None and elapsed_ms >= budget:
                break

            frac = (step - 1) / max(iters - 1, 1)
            temperature = t_start * ((t_end / t_start) ** frac)

            idx = int(rng.integers(0, n))
            candidate = x.copy()
            candidate[idx] = 1 - candidate[idx]
            candidate_value = evaluate_qubo(problem_ir, candidate)
            delta = candidate_value - value

            if delta <= 0 or rng.random() < np.exp(-delta / max(temperature, 1e-9)):
                x = candidate
                value = candidate_value
                if value < best_value:
                    best_value = value
                    best_x = x.copy()

            last_iter = step
            if step % emit_every == 0 or step == iters:
                events.append(
                    ProgressEvent(
                        t_ms=max(int((time.perf_counter() - started) * 1000.0), 0),
                        iter=step,
                        best_value=best_value,
                    )
                )

        runtime_ms = max(int((time.perf_counter() - started) * 1000.0), 0)
        if events[-1].iter != last_iter:
            events.append(ProgressEvent(t_ms=runtime_ms, iter=last_iter, best_value=best_value))

        solution = Solution(
            solver_name=self.name,
            best_x=best_x,
            best_value=best_value,
            metadata={
                "iters": iters,
                "seed": seed,
                "t_start": t_start,
                "t_end": t_end,
                "runtime_ms": runtime_ms,
                "completed_iters": last_iter,
            },
        )
        return solution, events

    def solve(self, problem_ir: ProblemIR, **kwargs: Any) -> Solution:
        solution, _ = self._run(problem_ir, **kwargs)
        return solution

    def solve_stream(self, problem_ir: ProblemIR, **kwargs: Any) -> Iterator[ProgressEvent]:
        _, events = self._run(problem_ir, **kwargs)
        yield from events


def get_solver() -> Solver:
    return SimulatedAnnealingSolver()
