"""Solution and status models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SolutionStatus(str, Enum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"


class Solution(BaseModel):
    status: SolutionStatus
    objective_value: float | None = None
    primal_values: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None

    def is_success(self) -> bool:
        return self.status in {SolutionStatus.OPTIMAL, SolutionStatus.FEASIBLE}
