from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from rastion.ir.types import ProblemIR
from rastion.solvers.base import ProgressEvent, Solution, Solver, evaluate_qubo


@dataclass
class TabuSolver(Solver):
    name: str = "tabu"
    supports: list[str] = None
    max_size: int = 200
    hardware: list[str] = None
    quality: float = 0.75

    def __post_init__(self) -> None:
        if self.supports is None:
            self.supports = ["qubo"]
        if self.hardware is None:
            self.hardware = ["cpu"]

    def _run(self, problem_ir: ProblemIR, **kwargs: Any) -> tuple[Solution, list[ProgressEvent]]:
        seed = int(kwargs.get("seed", 0))
        iters = int(kwargs.get("iters", 200))
        tenure = int(kwargs.get("tabu_tenure", 7))
        emit_every = max(int(kwargs.get("emit_every", 25)), 1)
        time_budget_ms = kwargs.get("time_budget_ms")
        budget = float(time_budget_ms) if time_budget_ms is not None else None

        started = time.perf_counter()
        n = problem_ir.n_vars

        x = np.where(problem_ir.objective.c < 0.0, 1, 0).astype(int)
        if seed:
            # Add deterministic noise from seed to diversify runs while remaining reproducible.
            rng = np.random.default_rng(seed)
            flip_mask = rng.integers(0, 2, size=n, dtype=int)
            x = np.bitwise_xor(x, flip_mask)

        best_x = x.copy()
        best_value = evaluate_qubo(problem_ir, best_x)
        tabu_until = np.zeros(n, dtype=int)

        events: list[ProgressEvent] = [ProgressEvent(t_ms=0, iter=0, best_value=best_value)]
        last_iter = 0

        for step in range(1, iters + 1):
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if budget is not None and elapsed_ms >= budget:
                break

            chosen_idx = None
            chosen_value = float("inf")
            chosen_vec = None

            for idx in range(n):
                trial = x.copy()
                trial[idx] = 1 - trial[idx]
                value = evaluate_qubo(problem_ir, trial)
                is_tabu = step < tabu_until[idx]
                aspiration = value < best_value
                if is_tabu and not aspiration:
                    continue
                if value < chosen_value:
                    chosen_value = value
                    chosen_vec = trial
                    chosen_idx = idx

            if chosen_vec is None:
                break

            x = chosen_vec
            tabu_until[chosen_idx] = step + tenure

            if chosen_value < best_value:
                best_value = chosen_value
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
                "tabu_tenure": tenure,
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
    return TabuSolver()
