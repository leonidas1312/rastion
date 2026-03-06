from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TSPProblem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    problem_type: Literal["tsp"] = "tsp"
    n_vars: int = Field(gt=2)
    depot: int = Field(default=0, ge=0)
    coords: np.ndarray
    edge_weight_type: str = "EUC_2D"
    source: str | None = None
    distance_matrix: np.ndarray | None = None

    @field_validator("coords", mode="before")
    @classmethod
    def _coerce_coords(cls, value: Any) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("coords must be an (n, 2) array")
        return arr

    @field_validator("distance_matrix", mode="before")
    @classmethod
    def _coerce_distance_matrix(cls, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("distance_matrix must be square")
        return arr

    @model_validator(mode="after")
    def _validate_structure(self) -> "TSPProblem":
        if self.coords.shape[0] != self.n_vars:
            raise ValueError("n_vars must match coordinate count")
        if self.depot >= self.n_vars:
            raise ValueError("depot index out of range")

        if self.distance_matrix is None:
            self.distance_matrix = _build_euc_2d_distance_matrix(self.coords)
        if self.distance_matrix.shape != (self.n_vars, self.n_vars):
            raise ValueError("distance matrix shape mismatch")
        return self


def _build_euc_2d_distance_matrix(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    d = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            delta = coords[i] - coords[j]
            dist = float(np.hypot(delta[0], delta[1]))
            tsp_dist = float(int(round(dist)))
            d[i, j] = tsp_dist
            d[j, i] = tsp_dist
    return d
