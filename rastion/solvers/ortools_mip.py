"""OR-Tools MIP solver adapter."""

from __future__ import annotations

import time

from rastion.core.ir import IRModel
from rastion.core.solution import Solution, SolutionStatus
from rastion.core.spec import ConstraintSense, ObjectiveSense, VariableType
from rastion.solvers.base import CapabilitySet, ConstraintType, ObjectiveType, SolveMode, SolverPlugin


class ORToolsMIPPlugin(SolverPlugin):
    name = "ortools"

    def __init__(self, version: str) -> None:
        self.version = version

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            variable_types={VariableType.BINARY, VariableType.INTEGER, VariableType.CONTINUOUS},
            objective_types={ObjectiveType.LINEAR},
            constraint_types={ConstraintType.LINEAR},
            modes={SolveMode.SOLVE},
            result_fields={"objective_value", "primal_values", "runtime_s"},
        )

    def solve(self, ir_model: IRModel, config: dict[str, object], backend: object) -> Solution:
        if ir_model.qubo is not None:
            raise ValueError("ortools plugin does not support QUBO target")
        if ir_model.objective.q_v:
            raise ValueError("ortools plugin does not support quadratic objective in MVP")

        from ortools.linear_solver import pywraplp

        backend_name = str(config.get("ortools_backend", "SCIP"))
        solver = pywraplp.Solver.CreateSolver(backend_name)
        if solver is None:
            solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
        if solver is None:
            raise RuntimeError("OR-Tools did not provide SCIP or CBC backend")

        time_limit = config.get("time_limit")
        if time_limit is not None:
            solver.SetTimeLimit(int(float(time_limit) * 1000))

        variables = []
        for var in ir_model.variables:
            if var.vartype == VariableType.BINARY:
                decision = solver.BoolVar(var.name)
            elif var.vartype == VariableType.INTEGER:
                decision = solver.IntVar(var.lb, var.ub, var.name)
            else:
                decision = solver.NumVar(var.lb, var.ub, var.name)
            variables.append(decision)

        rows: dict[int, list[tuple[int, float]]] = {}
        if ir_model.constraints is not None:
            for row, col, value in zip(
                ir_model.constraints.row_indices,
                ir_model.constraints.col_indices,
                ir_model.constraints.values,
                strict=True,
            ):
                rows.setdefault(row, []).append((col, value))

            for row in range(ir_model.constraints.num_rows):
                sense = ir_model.constraints.senses[row]
                rhs = ir_model.constraints.rhs[row]

                if sense == ConstraintSense.LE:
                    ct = solver.RowConstraint(-solver.infinity(), rhs, f"c_{row}")
                elif sense == ConstraintSense.GE:
                    ct = solver.RowConstraint(rhs, solver.infinity(), f"c_{row}")
                elif sense == ConstraintSense.EQ:
                    ct = solver.RowConstraint(rhs, rhs, f"c_{row}")
                else:
                    raise ValueError(f"unsupported constraint sense: {sense}")

                for col, coef in rows.get(row, []):
                    ct.SetCoefficient(variables[col], coef)

        objective = solver.Objective()
        for idx, coef in enumerate(ir_model.objective.linear):
            if coef:
                objective.SetCoefficient(variables[idx], coef)

        if ir_model.objective.sense == ObjectiveSense.MAX:
            objective.SetMaximization()
        else:
            objective.SetMinimization()

        start = time.perf_counter()
        status_code = solver.Solve()
        runtime_s = time.perf_counter() - start

        status_map = {
            pywraplp.Solver.OPTIMAL: SolutionStatus.OPTIMAL,
            pywraplp.Solver.FEASIBLE: SolutionStatus.FEASIBLE,
            pywraplp.Solver.INFEASIBLE: SolutionStatus.INFEASIBLE,
            pywraplp.Solver.UNBOUNDED: SolutionStatus.UNKNOWN,
            pywraplp.Solver.ABNORMAL: SolutionStatus.ERROR,
            pywraplp.Solver.NOT_SOLVED: SolutionStatus.UNKNOWN,
        }
        status = status_map.get(status_code, SolutionStatus.UNKNOWN)

        primal_values: dict[str, float] = {}
        objective_value: float | None = None
        if status in {SolutionStatus.OPTIMAL, SolutionStatus.FEASIBLE}:
            primal_values = {
                ir_model.variables[i].name: float(variables[i].solution_value())
                for i in range(len(ir_model.variables))
            }
            objective_value = float(objective.Value())

        return Solution(
            status=status,
            objective_value=objective_value,
            primal_values=primal_values,
            metadata={
                "runtime_s": runtime_s,
                "solver_name": self.name,
                "solver_version": self.version,
            },
            error_message=None if status != SolutionStatus.ERROR else "OR-Tools returned ABNORMAL",
        )


def get_plugin() -> ORToolsMIPPlugin | None:
    try:
        import ortools
    except ImportError:
        return None
    version = getattr(ortools, "__version__", "unknown")
    return ORToolsMIPPlugin(version=version)
