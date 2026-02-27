"""PySCIPOpt solver adapter."""

from __future__ import annotations

import time

from rastion.core.ir import IRModel
from rastion.core.solution import Solution, SolutionStatus
from rastion.core.spec import ConstraintSense, ObjectiveSense, VariableType
from rastion.solvers.base import CapabilitySet, ConstraintType, ObjectiveType, SolveMode, SolverPlugin


class SCIPPlugin(SolverPlugin):
    name = "scip"

    def __init__(self, version: str) -> None:
        self.version = version

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            variable_types={VariableType.BINARY, VariableType.INTEGER, VariableType.CONTINUOUS},
            objective_types={ObjectiveType.LINEAR},
            constraint_types={ConstraintType.LINEAR},
            modes={SolveMode.SOLVE},
            result_fields={"objective_value", "primal_values", "runtime_s", "gap"},
        )

    def solve(self, ir_model: IRModel, config: dict[str, object], backend: object) -> Solution:
        if ir_model.qubo is not None:
            raise ValueError("scip plugin does not support QUBO target")
        if ir_model.objective.q_v:
            raise ValueError("scip plugin does not support quadratic objective in this MVP")

        from pyscipopt import Model, quicksum

        model = Model("rastion")

        time_limit = config.get("time_limit")
        if time_limit is not None:
            model.setParam("limits/time", float(time_limit))
        mip_gap = config.get("mip_gap")
        if mip_gap is not None:
            model.setParam("limits/gap", float(mip_gap))
        seed = config.get("seed")
        if seed is not None:
            model.setParam("randomization/randomseedshift", int(seed))

        vars_by_index = []
        for var in ir_model.variables:
            if var.vartype == VariableType.BINARY:
                vtype = "B"
            elif var.vartype == VariableType.INTEGER:
                vtype = "I"
            else:
                vtype = "C"
            decision = model.addVar(name=var.name, vtype=vtype, lb=var.lb, ub=var.ub)
            vars_by_index.append(decision)

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
                expr = quicksum(coef * vars_by_index[col] for col, coef in rows.get(row, []))
                rhs = ir_model.constraints.rhs[row]
                sense = ir_model.constraints.senses[row]

                if sense == ConstraintSense.LE:
                    model.addCons(expr <= rhs, name=f"c_{row}")
                elif sense == ConstraintSense.GE:
                    model.addCons(expr >= rhs, name=f"c_{row}")
                elif sense == ConstraintSense.EQ:
                    model.addCons(expr == rhs, name=f"c_{row}")
                else:
                    raise ValueError(f"unsupported constraint sense: {sense}")

        objective_expr = quicksum(
            coef * vars_by_index[idx] for idx, coef in enumerate(ir_model.objective.linear) if coef
        )
        if ir_model.objective.sense == ObjectiveSense.MAX:
            model.setObjective(objective_expr, "maximize")
        else:
            model.setObjective(objective_expr, "minimize")

        start = time.perf_counter()
        model.optimize()
        runtime_s = time.perf_counter() - start

        status_text = str(model.getStatus()).lower()
        if "optimal" in status_text:
            status = SolutionStatus.OPTIMAL
        elif "infeasible" in status_text:
            status = SolutionStatus.INFEASIBLE
        elif "timelimit" in status_text or "gaplimit" in status_text or "bestsollimit" in status_text:
            status = SolutionStatus.FEASIBLE if model.getNSols() > 0 else SolutionStatus.UNKNOWN
        elif "error" in status_text:
            status = SolutionStatus.ERROR
        else:
            status = SolutionStatus.UNKNOWN

        primal_values: dict[str, float] = {}
        objective_value: float | None = None
        gap: float | None = None

        if model.getNSols() > 0 and status in {SolutionStatus.OPTIMAL, SolutionStatus.FEASIBLE, SolutionStatus.UNKNOWN}:
            solution = model.getBestSol()
            if solution is not None:
                primal_values = {
                    ir_model.variables[i].name: float(model.getSolVal(solution, vars_by_index[i]))
                    for i in range(len(vars_by_index))
                }
                objective_value = float(model.getObjVal())
                try:
                    gap = float(model.getGap())
                except Exception:
                    gap = None
                if status == SolutionStatus.UNKNOWN:
                    status = SolutionStatus.FEASIBLE

        return Solution(
            status=status,
            objective_value=objective_value,
            primal_values=primal_values,
            metadata={
                "runtime_s": runtime_s,
                "solver_name": self.name,
                "solver_version": self.version,
                "gap": gap,
            },
            error_message=None if status != SolutionStatus.ERROR else "SCIP returned an error status",
        )


def get_plugin() -> SCIPPlugin | None:
    try:
        import pyscipopt
    except ImportError:
        return None
    version = getattr(pyscipopt, "__version__", "unknown")
    return SCIPPlugin(version=version)
