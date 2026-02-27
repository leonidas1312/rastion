"""Schema models for optimization problem structure."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

CURRENT_SCHEMA_VERSION = "0.1.0"


class VariableType(str, Enum):
    BINARY = "binary"
    INTEGER = "integer"
    CONTINUOUS = "continuous"


class ObjectiveSense(str, Enum):
    MIN = "min"
    MAX = "max"


class ConstraintSense(str, Enum):
    LE = "<="
    GE = ">="
    EQ = "=="


class IRTarget(str, Enum):
    GENERIC = "generic"
    QUBO = "qubo"


class VariableSpec(BaseModel):
    name: str
    vartype: VariableType
    lb: float | None = None
    ub: float | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> "VariableSpec":
        if self.vartype == VariableType.BINARY:
            if self.lb is None:
                self.lb = 0.0
            if self.ub is None:
                self.ub = 1.0
            if self.lb != 0 or self.ub != 1:
                raise ValueError(f"binary variable '{self.name}' must have lb=0 and ub=1")
            return self

        if self.lb is None or self.ub is None:
            raise ValueError(f"variable '{self.name}' must define lb and ub")
        if self.lb > self.ub:
            raise ValueError(f"variable '{self.name}' has lb > ub")

        if self.vartype == VariableType.INTEGER:
            if not float(self.lb).is_integer() or not float(self.ub).is_integer():
                raise ValueError(f"integer variable '{self.name}' must have integral bounds")
        return self


class ObjectiveSpec(BaseModel):
    sense: ObjectiveSense = ObjectiveSense.MIN
    linear: str | None = None
    quadratic: str | None = None
    constant: float = 0.0


class ConstraintBlockSpec(BaseModel):
    name: str
    matrix: str
    rhs: str
    sense: ConstraintSense


class ProblemSpec(BaseModel):
    schema_version: str = CURRENT_SCHEMA_VERSION
    name: str = "unnamed_problem"
    ir_target: IRTarget = IRTarget.GENERIC
    variables: list[VariableSpec] = Field(default_factory=list)
    objective: ObjectiveSpec
    constraints: list[ConstraintBlockSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_model(self) -> "ProblemSpec":
        ensure_schema_compatible(self.schema_version, "ProblemSpec")
        if not self.variables:
            raise ValueError("at least one variable is required")

        names = [v.name for v in self.variables]
        if len(names) != len(set(names)):
            raise ValueError("variable names must be unique")

        if self.ir_target == IRTarget.QUBO:
            non_binary = [v.name for v in self.variables if v.vartype != VariableType.BINARY]
            if non_binary:
                raise ValueError(
                    "QUBO target requires binary variables only; invalid: "
                    + ", ".join(non_binary)
                )
            if self.constraints:
                raise ValueError("QUBO target does not allow linear constraints in this MVP")
        return self

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ProblemSpec":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def to_json_file(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, sort_keys=True)


def is_schema_compatible(version: str) -> bool:
    try:
        major = int(version.split(".", 1)[0])
        current_major = int(CURRENT_SCHEMA_VERSION.split(".", 1)[0])
    except (TypeError, ValueError, IndexError):
        return False
    return major == current_major


def ensure_schema_compatible(version: str, kind: str) -> None:
    if not is_schema_compatible(version):
        raise ValueError(
            f"{kind} schema_version '{version}' is incompatible with "
            f"supported '{CURRENT_SCHEMA_VERSION}'"
        )
