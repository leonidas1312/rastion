from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Variable(BaseModel):
    name: str
    domain: Literal["binary"] = "binary"


class Objective(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    Q: np.ndarray
    c: np.ndarray
    constant: float = 0.0

    @field_validator("Q", mode="before")
    @classmethod
    def _coerce_q(cls, value: Any) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Q must be a 2D matrix")
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("Q must be square")
        return arr

    @field_validator("c", mode="before")
    @classmethod
    def _coerce_c(cls, value: Any) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 1:
            raise ValueError("c must be a 1D vector")
        return arr

    @model_validator(mode="after")
    def _match_dimensions(self) -> "Objective":
        if self.Q.shape[0] != self.c.shape[0]:
            raise ValueError("Q and c dimensions must match")
        return self


class ProblemIR(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    problem_type: Literal["qubo"]
    n_vars: int = Field(gt=0)
    variables: list[Variable]
    objective: Objective
    constraints: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def _validate_structure(self) -> "ProblemIR":
        if len(self.variables) != self.n_vars:
            raise ValueError("n_vars must match number of variables")
        if self.objective.Q.shape != (self.n_vars, self.n_vars):
            raise ValueError("Q shape must be (n_vars, n_vars)")
        if self.objective.c.shape != (self.n_vars,):
            raise ValueError("c shape must be (n_vars,)")
        return self
