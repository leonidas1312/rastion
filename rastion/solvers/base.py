from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rastion.ir.types import ProblemIR


class Solution(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    solver_name: str
    best_x: np.ndarray
    best_value: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    selection_reason: str | None = None

    @field_validator("best_x", mode="before")
    @classmethod
    def _coerce_best_x(cls, value: Any) -> np.ndarray:
        arr = np.asarray(value, dtype=int)
        if arr.ndim != 1:
            raise ValueError("best_x must be a 1D vector")
        return arr


class ProgressEvent(BaseModel):
    t_ms: int = Field(ge=0)
    iter: int = Field(ge=0)
    best_value: float


class Solver(ABC):
    name: str
    supports: list[str]
    max_size: int
    hardware: list[str]
    quality: float = 0.5

    @abstractmethod
    def solve(self, problem_ir: ProblemIR, **kwargs: Any) -> Solution:
        raise NotImplementedError

    def solve_trace(self, problem_ir: ProblemIR, **kwargs: Any) -> tuple[Solution, list[ProgressEvent]]:
        solution = self.solve(problem_ir, **kwargs)
        runtime_ms = solution.metadata.get("runtime_ms", 0)
        final_event = ProgressEvent(
            t_ms=max(int(runtime_ms), 0),
            iter=int(kwargs.get("iters", 1)),
            best_value=solution.best_value,
        )
        return solution, [final_event]

    def solve_stream(self, problem_ir: ProblemIR, **kwargs: Any) -> Iterator[ProgressEvent]:
        _, events = self.solve_trace(problem_ir, **kwargs)
        yield from events


def evaluate_qubo(problem_ir: ProblemIR, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    q = problem_ir.objective.Q
    c = problem_ir.objective.c
    return float(x.T @ q @ x + c.T @ x + problem_ir.objective.constant)
