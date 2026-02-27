"""HiGHS (highspy) solver adapter."""

from __future__ import annotations

import time

import numpy as np

from rastion.core.ir import IRModel
from rastion.core.solution import Solution, SolutionStatus
from rastion.core.spec import ObjectiveSense, VariableType
from rastion.solvers.base import CapabilitySet, ConstraintType, ObjectiveType, SolveMode, SolverPlugin


class HighsPlugin(SolverPlugin):
    name = "highs"

    def __init__(self, version: str) -> None:
        self.version = version

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            variable_types={VariableType.BINARY, VariableType.INTEGER, VariableType.CONTINUOUS},
            objective_types={ObjectiveType.LINEAR, ObjectiveType.QUADRATIC},
            constraint_types={ConstraintType.LINEAR},
            modes={SolveMode.SOLVE},
            result_fields={"objective_value", "primal_values", "runtime_s", "gap"},
            supports_miqp=False,
        )

    def solve(self, ir_model: IRModel, config: dict[str, object], backend: object) -> Solution:
        if ir_model.qubo is not None:
            raise ValueError("highs plugin does not support QUBO target directly")

        import highspy

        has_integer = any(v.vartype in {VariableType.BINARY, VariableType.INTEGER} for v in ir_model.variables)
        has_quadratic = bool(ir_model.objective.q_v)
        if has_integer and has_quadratic:
            raise ValueError("HiGHS does not support mixed-integer quadratic objective")

        model = self._build_model(highspy, ir_model)

        highs = highspy.Highs()
        time_limit = config.get("time_limit")
        if time_limit is not None:
            highs.setOptionValue("time_limit", float(time_limit))
        mip_gap = config.get("mip_gap")
        if mip_gap is not None:
            highs.setOptionValue("mip_rel_gap", float(mip_gap))
        seed = config.get("seed")
        if seed is not None:
            highs.setOptionValue("random_seed", int(seed))

        pass_status = highs.passModel(model)
        if str(pass_status).lower().find("error") >= 0:
            raise RuntimeError(f"HiGHS failed to load model: {pass_status}")

        start = time.perf_counter()
        highs.run()
        runtime_s = time.perf_counter() - start

        status_text = highs.modelStatusToString(highs.getModelStatus()).lower()
        if "optimal" in status_text:
            status = SolutionStatus.OPTIMAL
        elif "infeasible" in status_text:
            status = SolutionStatus.INFEASIBLE
        elif "unbounded" in status_text:
            status = SolutionStatus.UNKNOWN
        elif "error" in status_text:
            status = SolutionStatus.ERROR
        else:
            status = SolutionStatus.UNKNOWN

        solution = highs.getSolution()
        info = highs.getInfo()

        primal_values: dict[str, float] = {}
        objective_value: float | None = None
        if status in {SolutionStatus.OPTIMAL, SolutionStatus.FEASIBLE, SolutionStatus.UNKNOWN}:
            try:
                col_values = list(solution.col_value)
            except Exception:
                col_values = []
            if col_values:
                primal_values = {
                    ir_model.variables[i].name: float(col_values[i])
                    for i in range(len(ir_model.variables))
                }
                if status == SolutionStatus.UNKNOWN:
                    status = SolutionStatus.FEASIBLE

            try:
                objective_value = float(info.objective_function_value)
            except Exception:
                objective_value = None

        if objective_value is not None and ir_model.objective.sense == ObjectiveSense.MAX:
            objective_value = -objective_value

        gap = None
        try:
            gap = float(info.mip_gap)
        except Exception:
            gap = None

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
            error_message=None if status != SolutionStatus.ERROR else "HiGHS returned an error status",
        )

    def _build_model(self, highspy: object, ir_model: IRModel) -> object:
        n = len(ir_model.variables)
        m = ir_model.constraints.num_rows if ir_model.constraints is not None else 0

        lp = highspy.HighsLp()
        lp.num_col_ = n
        lp.num_row_ = m

        obj_sign = 1.0 if ir_model.objective.sense == ObjectiveSense.MIN else -1.0

        lp.col_cost_ = np.asarray([obj_sign * c for c in ir_model.objective.linear], dtype=np.double)
        lp.col_lower_ = np.asarray([v.lb for v in ir_model.variables], dtype=np.double)
        lp.col_upper_ = np.asarray([v.ub for v in ir_model.variables], dtype=np.double)
        lp.offset_ = obj_sign * float(ir_model.objective.constant)

        if m > 0 and ir_model.constraints is not None:
            row_lower = np.full(m, -highspy.kHighsInf, dtype=np.double)
            row_upper = np.full(m, highspy.kHighsInf, dtype=np.double)
            for row in range(m):
                sense = ir_model.constraints.senses[row]
                rhs = ir_model.constraints.rhs[row]
                if sense.value == "<=":
                    row_upper[row] = rhs
                elif sense.value == ">=":
                    row_lower[row] = rhs
                elif sense.value == "==":
                    row_lower[row] = rhs
                    row_upper[row] = rhs
            lp.row_lower_ = row_lower
            lp.row_upper_ = row_upper
        else:
            lp.row_lower_ = np.asarray([], dtype=np.double)
            lp.row_upper_ = np.asarray([], dtype=np.double)

        col_entries: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        if ir_model.constraints is not None:
            for row, col, value in zip(
                ir_model.constraints.row_indices,
                ir_model.constraints.col_indices,
                ir_model.constraints.values,
                strict=True,
            ):
                col_entries[col].append((row, float(value)))

        starts = [0]
        index: list[int] = []
        values: list[float] = []
        for col in range(n):
            for row, val in sorted(col_entries[col]):
                index.append(row)
                values.append(val)
            starts.append(len(index))

        lp.a_matrix_.num_col_ = n
        lp.a_matrix_.num_row_ = m
        lp.a_matrix_.start_ = np.asarray(starts, dtype=np.int64)
        lp.a_matrix_.index_ = np.asarray(index, dtype=np.int32)
        lp.a_matrix_.value_ = np.asarray(values, dtype=np.double)

        if hasattr(highspy, "MatrixFormat") and hasattr(highspy.MatrixFormat, "kColwise"):
            lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise

        if any(v.vartype != VariableType.CONTINUOUS for v in ir_model.variables):
            integrality = []
            for var in ir_model.variables:
                if var.vartype == VariableType.CONTINUOUS:
                    integrality.append(highspy.HighsVarType.kContinuous)
                else:
                    integrality.append(highspy.HighsVarType.kInteger)
            lp.integrality_ = integrality

        if not ir_model.objective.q_v:
            return lp

        if not hasattr(highspy, "HighsModel"):
            raise ValueError("highspy build lacks HighsModel support for quadratic objectives")

        qp = highspy.HighsModel()
        qp.lp_ = lp

        hessian = highspy.HighsHessian()
        hessian.dim_ = n
        if hasattr(highspy, "HessianFormat") and hasattr(highspy.HessianFormat, "kTriangular"):
            hessian.format_ = highspy.HessianFormat.kTriangular

        col_hess: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for i, j, coeff in zip(ir_model.objective.q_i, ir_model.objective.q_j, ir_model.objective.q_v, strict=True):
            scaled = obj_sign * float(coeff)
            if i == j:
                col_hess[j].append((i, 2.0 * scaled))
            else:
                row, col = (i, j) if i <= j else (j, i)
                col_hess[col].append((row, scaled))

        h_start = [0]
        h_index: list[int] = []
        h_value: list[float] = []
        for col in range(n):
            for row, val in sorted(col_hess[col]):
                if val != 0:
                    h_index.append(row)
                    h_value.append(val)
            h_start.append(len(h_index))

        hessian.start_ = np.asarray(h_start, dtype=np.int64)
        hessian.index_ = np.asarray(h_index, dtype=np.int32)
        hessian.value_ = np.asarray(h_value, dtype=np.double)
        qp.hessian_ = hessian
        return qp


def get_plugin() -> HighsPlugin | None:
    try:
        import highspy
    except ImportError:
        return None
    version = getattr(highspy, "HIGHS_VERSION_MAJOR", None)
    if version is None:
        version_text = getattr(highspy, "__version__", "unknown")
    else:
        major = getattr(highspy, "HIGHS_VERSION_MAJOR", "")
        minor = getattr(highspy, "HIGHS_VERSION_MINOR", "")
        patch = getattr(highspy, "HIGHS_VERSION_PATCH", "")
        version_text = f"{major}.{minor}.{patch}"
    return HighsPlugin(version=version_text)
