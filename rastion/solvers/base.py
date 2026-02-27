"""Solver plugin base classes and capability schemas."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel, Field

from rastion.core.ir import IRModel
from rastion.core.solution import Solution
from rastion.core.spec import VariableType


class ObjectiveType(str, Enum):
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    QUBO = "qubo"


class ConstraintType(str, Enum):
    LINEAR = "linear"


class SolveMode(str, Enum):
    SOLVE = "solve"
    SAMPLE = "sample"


class CapabilitySet(BaseModel):
    variable_types: set[VariableType] = Field(default_factory=set)
    objective_types: set[ObjectiveType] = Field(default_factory=set)
    constraint_types: set[ConstraintType] = Field(default_factory=set)
    modes: set[SolveMode] = Field(default_factory=lambda: {SolveMode.SOLVE})
    result_fields: set[str] = Field(default_factory=set)
    supports_miqp: bool = False


class ProblemRequirements(BaseModel):
    variable_types: set[VariableType] = Field(default_factory=set)
    objective_type: ObjectiveType
    constraint_types: set[ConstraintType] = Field(default_factory=set)
    mode: SolveMode = SolveMode.SOLVE
    has_integer_variables: bool = False


class SolverPlugin(ABC):
    name: str
    version: str

    @abstractmethod
    def capabilities(self) -> CapabilitySet:
        raise NotImplementedError

    @abstractmethod
    def solve(self, ir_model: IRModel, config: dict[str, object], backend: object) -> Solution:
        raise NotImplementedError
