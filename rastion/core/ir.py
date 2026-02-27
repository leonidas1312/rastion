"""Normalized internal representation (IR)."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from rastion.core.spec import ConstraintSense, IRTarget, ObjectiveSense, VariableType


class IRVariable(BaseModel):
    index: int
    name: str
    vartype: VariableType
    lb: float
    ub: float


class IRObjective(BaseModel):
    sense: ObjectiveSense
    linear: list[float]
    q_i: list[int] = Field(default_factory=list)
    q_j: list[int] = Field(default_factory=list)
    q_v: list[float] = Field(default_factory=list)
    constant: float = 0.0

    @model_validator(mode="after")
    def _validate_shapes(self) -> "IRObjective":
        if len(self.q_i) != len(self.q_j) or len(self.q_i) != len(self.q_v):
            raise ValueError("quadratic term arrays q_i, q_j, q_v must have equal length")
        return self


class IRConstraintMatrix(BaseModel):
    num_rows: int
    senses: list[ConstraintSense]
    rhs: list[float]
    row_indices: list[int] = Field(default_factory=list)
    col_indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_shapes(self) -> "IRConstraintMatrix":
        if len(self.senses) != self.num_rows or len(self.rhs) != self.num_rows:
            raise ValueError("constraint senses and rhs must match num_rows")
        if len(self.row_indices) != len(self.col_indices) or len(self.row_indices) != len(self.values):
            raise ValueError("constraint COO arrays row_indices, col_indices, values must match")
        return self


class QUBOIR(BaseModel):
    n: int
    q_i: list[int] = Field(default_factory=list)
    q_j: list[int] = Field(default_factory=list)
    q_v: list[float] = Field(default_factory=list)
    linear: list[float] = Field(default_factory=list)
    constant: float = 0.0

    @model_validator(mode="after")
    def _validate_shapes(self) -> "QUBOIR":
        if len(self.q_i) != len(self.q_j) or len(self.q_i) != len(self.q_v):
            raise ValueError("QUBO arrays q_i, q_j, q_v must have equal length")
        if len(self.linear) != self.n:
            raise ValueError("QUBO linear vector length must equal n")
        return self


class IRModel(BaseModel):
    schema_version: str
    target: IRTarget = IRTarget.GENERIC
    variables: list[IRVariable]
    objective: IRObjective
    constraints: IRConstraintMatrix | None = None
    qubo: QUBOIR | None = None

    @model_validator(mode="after")
    def _validate_shapes(self) -> "IRModel":
        n = len(self.variables)
        if len(self.objective.linear) != n:
            raise ValueError("objective linear vector length must match variable count")
        if self.qubo is not None and self.qubo.n != n:
            raise ValueError("QUBO variable count must match variable count")
        return self

    def variable_names(self) -> list[str]:
        return [v.name for v in self.variables]

    def variable_types(self) -> set[VariableType]:
        return {v.vartype for v in self.variables}

    def has_constraints(self) -> bool:
        return self.constraints is not None and self.constraints.num_rows > 0

    def has_quadratic_objective(self) -> bool:
        return bool(self.objective.q_v)
