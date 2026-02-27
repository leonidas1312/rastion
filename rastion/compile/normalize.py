"""Compilation from schema + instance data into normalized IR."""

from __future__ import annotations

import numpy as np

from rastion.compile.qubo import objective_to_qubo
from rastion.core.data import InstanceData
from rastion.core.ir import IRConstraintMatrix, IRModel, IRObjective, IRVariable
from rastion.core.spec import ProblemSpec
from rastion.core.validate import validate_problem_and_instance


def compile_to_ir(spec: ProblemSpec, instance: InstanceData) -> IRModel:
    result = validate_problem_and_instance(spec, instance)
    if not result.valid:
        msg = "Invalid problem/instance: " + "; ".join(result.errors)
        raise ValueError(msg)

    n = len(spec.variables)
    variables: list[IRVariable] = []
    for idx, var in enumerate(spec.variables):
        assert var.lb is not None
        assert var.ub is not None
        variables.append(
            IRVariable(
                index=idx,
                name=var.name,
                vartype=var.vartype,
                lb=float(var.lb),
                ub=float(var.ub),
            )
        )

    linear = np.zeros(n, dtype=float)
    if spec.objective.linear:
        linear = np.asarray(instance.get_array(spec.objective.linear, ndim=1), dtype=float)

    q_i: list[int] = []
    q_j: list[int] = []
    q_v: list[float] = []
    if spec.objective.quadratic:
        matrix = np.asarray(instance.get_array(spec.objective.quadratic, ndim=2), dtype=float)
        for i in range(n):
            for j in range(i, n):
                coeff = float(matrix[i, j])
                if i != j:
                    coeff += float(matrix[j, i])
                if coeff != 0.0:
                    q_i.append(i)
                    q_j.append(j)
                    q_v.append(coeff)

    objective = IRObjective(
        sense=spec.objective.sense,
        linear=linear.tolist(),
        q_i=q_i,
        q_j=q_j,
        q_v=q_v,
        constant=float(spec.objective.constant),
    )

    constraints: IRConstraintMatrix | None = None
    if spec.constraints:
        row_indices: list[int] = []
        col_indices: list[int] = []
        values: list[float] = []
        senses: list = []
        rhs: list[float] = []

        global_row = 0
        for block in spec.constraints:
            a = np.asarray(instance.get_array(block.matrix, ndim=2), dtype=float)
            b = np.asarray(instance.get_array(block.rhs, ndim=1), dtype=float)

            for local_row in range(a.shape[0]):
                senses.append(block.sense)
                rhs.append(float(b[local_row]))

                nz_cols = np.nonzero(a[local_row])[0]
                for col in nz_cols:
                    row_indices.append(global_row)
                    col_indices.append(int(col))
                    values.append(float(a[local_row, col]))

                global_row += 1

        constraints = IRConstraintMatrix(
            num_rows=len(rhs),
            senses=senses,
            rhs=rhs,
            row_indices=row_indices,
            col_indices=col_indices,
            values=values,
        )

    qubo = objective_to_qubo(objective, n=n) if spec.ir_target.value == "qubo" else None

    return IRModel(
        schema_version=spec.schema_version,
        target=spec.ir_target,
        variables=variables,
        objective=objective,
        constraints=constraints,
        qubo=qubo,
    )
