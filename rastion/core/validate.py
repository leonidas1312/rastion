"""Validation helpers for ProblemSpec + InstanceData."""

from __future__ import annotations

from pydantic import BaseModel, Field

from rastion.core.data import InstanceData
from rastion.core.spec import IRTarget, ProblemSpec, ensure_schema_compatible


class ValidationResult(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def validate_problem_and_instance(spec: ProblemSpec, instance: InstanceData) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        ensure_schema_compatible(spec.schema_version, "ProblemSpec")
    except ValueError as exc:
        errors.append(str(exc))

    try:
        ensure_schema_compatible(instance.schema_version, "InstanceData")
    except ValueError as exc:
        errors.append(str(exc))

    n = len(spec.variables)

    if spec.objective.linear:
        try:
            linear = instance.get_array(spec.objective.linear, ndim=1)
            if linear.shape[0] != n:
                errors.append(
                    f"objective linear '{spec.objective.linear}' has length {linear.shape[0]}, expected {n}"
                )
        except (KeyError, ValueError) as exc:
            errors.append(str(exc))

    if spec.objective.quadratic:
        try:
            quad = instance.get_array(spec.objective.quadratic, ndim=2)
            if quad.shape != (n, n):
                errors.append(
                    f"objective quadratic '{spec.objective.quadratic}' has shape {quad.shape}, expected {(n, n)}"
                )
        except (KeyError, ValueError) as exc:
            errors.append(str(exc))

    for block in spec.constraints:
        try:
            a = instance.get_array(block.matrix, ndim=2)
        except (KeyError, ValueError) as exc:
            errors.append(f"constraint '{block.name}' matrix error: {exc}")
            continue

        if a.shape[1] != n:
            errors.append(
                f"constraint '{block.name}' matrix '{block.matrix}' has {a.shape[1]} columns, expected {n}"
            )

        try:
            b = instance.get_array(block.rhs, ndim=1)
        except (KeyError, ValueError) as exc:
            errors.append(f"constraint '{block.name}' rhs error: {exc}")
            continue

        if b.shape[0] != a.shape[0]:
            errors.append(
                f"constraint '{block.name}' rhs '{block.rhs}' length {b.shape[0]} "
                f"does not match rows {a.shape[0]}"
            )

    if spec.ir_target == IRTarget.QUBO and spec.constraints:
        errors.append("QUBO target does not support linear constraints in this MVP")

    if spec.ir_target == IRTarget.QUBO and spec.objective.quadratic is None and spec.objective.linear is None:
        warnings.append("QUBO target objective has no linear or quadratic terms")

    return ValidationResult(valid=not errors, errors=errors, warnings=warnings)
