from __future__ import annotations

import random
import time

from rastion.core.ir import IRModel
from rastion.core.solution import Solution, SolutionStatus
from rastion.core.spec import ConstraintSense, ObjectiveSense, VariableType
from rastion.solvers.base import CapabilitySet, ConstraintType, ObjectiveType, SolveMode, SolverPlugin


class GraspBinaryProSolver(SolverPlugin):
    """GRASP + repair + hill climbing for binary linear models."""

    name = "grasp_binary_pro"
    version = "0.1.0"

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            variable_types={VariableType.BINARY},
            objective_types={ObjectiveType.LINEAR},
            constraint_types={ConstraintType.LINEAR},
            modes={SolveMode.SOLVE},
            result_fields={"objective_value", "primal_values", "runtime_s"},
        )

    def solve(self, ir_model: IRModel, config: dict[str, object], backend: object) -> Solution:
        if ir_model.qubo is not None:
            raise ValueError("grasp_binary_pro does not support QUBO models")
        if ir_model.objective.q_v:
            raise ValueError("grasp_binary_pro supports linear objective only")
        if any(var.vartype != VariableType.BINARY for var in ir_model.variables):
            raise ValueError("grasp_binary_pro supports binary variables only")

        start = time.perf_counter()
        n = len(ir_model.variables)
        row_terms = self._row_terms(ir_model)

        seed = int(config.get("seed", 0))
        rng = random.Random(seed)

        time_limit = float(config.get("time_limit", 0.0)) if "time_limit" in config else None
        deadline = (start + time_limit) if time_limit and time_limit > 0 else None

        if n <= 22:
            best_x, best_obj, complete = self._bruteforce(ir_model, row_terms, deadline)
            runtime_s = time.perf_counter() - start
            if best_x is None:
                return Solution(
                    status=SolutionStatus.INFEASIBLE if complete else SolutionStatus.UNKNOWN,
                    objective_value=None,
                    primal_values={},
                    metadata={
                        "runtime_s": runtime_s,
                        "solver_name": self.name,
                        "solver_version": self.version,
                        "method": "bruteforce",
                    },
                )

            return Solution(
                status=SolutionStatus.OPTIMAL if complete else SolutionStatus.FEASIBLE,
                objective_value=best_obj,
                primal_values=self._to_named_solution(ir_model, best_x),
                metadata={
                    "runtime_s": runtime_s,
                    "solver_name": self.name,
                    "solver_version": self.version,
                    "method": "bruteforce",
                },
            )

        iterations_default = max(40, min(420, 5 * n))
        iterations = int(config.get("iterations", iterations_default))
        rcl_size = max(2, int(config.get("rcl_size", 8)))

        best_x: list[int] | None = None
        best_obj: float | None = None

        for _ in range(max(1, iterations)):
            if deadline is not None and time.perf_counter() >= deadline:
                break

            x = self._greedy_randomized_start(ir_model, row_terms, rng, rcl_size)
            x = self._repair_to_feasible(ir_model, row_terms, x, deadline)
            if x is None:
                continue

            x = self._local_search(ir_model, row_terms, x, rng, deadline)
            candidate = self._objective(ir_model, x)
            if self._is_better(ir_model, candidate, best_obj):
                best_x = list(x)
                best_obj = candidate

        runtime_s = time.perf_counter() - start
        if best_x is None:
            return Solution(
                status=SolutionStatus.UNKNOWN,
                objective_value=None,
                primal_values={},
                metadata={
                    "runtime_s": runtime_s,
                    "solver_name": self.name,
                    "solver_version": self.version,
                    "method": "grasp",
                    "iterations": iterations,
                },
            )

        return Solution(
            status=SolutionStatus.FEASIBLE,
            objective_value=best_obj,
            primal_values=self._to_named_solution(ir_model, best_x),
            metadata={
                "runtime_s": runtime_s,
                "solver_name": self.name,
                "solver_version": self.version,
                "method": "grasp",
                "iterations": iterations,
                "rcl_size": rcl_size,
            },
        )

    def _bruteforce(
        self,
        ir_model: IRModel,
        row_terms: list[list[tuple[int, float]]],
        deadline: float | None,
    ) -> tuple[list[int] | None, float | None, bool]:
        n = len(ir_model.variables)
        total = 1 << n
        best_x: list[int] | None = None
        best_obj: float | None = None
        complete = True

        for mask in range(total):
            if deadline is not None and mask % 4096 == 0 and time.perf_counter() >= deadline:
                complete = False
                break

            x = [(mask >> i) & 1 for i in range(n)]
            if not self._is_feasible(ir_model, row_terms, x):
                continue

            value = self._objective(ir_model, x)
            if self._is_better(ir_model, value, best_obj):
                best_x = x
                best_obj = value

        return best_x, best_obj, complete

    def _greedy_randomized_start(
        self,
        ir_model: IRModel,
        row_terms: list[list[tuple[int, float]]],
        rng: random.Random,
        rcl_size: int,
    ) -> list[int]:
        n = len(ir_model.variables)
        x = [0] * n
        pressure = self._constraint_pressure(ir_model, row_terms)

        while True:
            candidates: list[tuple[float, int]] = []
            for idx in range(n):
                if x[idx] == 1:
                    continue

                coeff = float(ir_model.objective.linear[idx])
                directional_gain = coeff if ir_model.objective.sense == ObjectiveSense.MAX else -coeff
                if directional_gain <= 0:
                    continue

                x[idx] = 1
                feasible = self._respects_upper_bounds(ir_model, row_terms, x)
                x[idx] = 0
                if not feasible:
                    continue

                score = directional_gain / (1.0 + pressure[idx])
                candidates.append((score, idx))

            if not candidates:
                break

            candidates.sort(reverse=True)
            selected_pool = candidates[: min(rcl_size, len(candidates))]
            _, chosen = rng.choice(selected_pool)
            x[chosen] = 1

            if rng.random() < 0.08:
                break

        return x

    def _repair_to_feasible(
        self,
        ir_model: IRModel,
        row_terms: list[list[tuple[int, float]]],
        x: list[int],
        deadline: float | None,
    ) -> list[int] | None:
        n = len(x)
        for _ in range(max(30, 8 * n)):
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            if self._is_feasible(ir_model, row_terms, x):
                return x

            current_violation = self._violation(ir_model, row_terms, x)
            best_idx = -1
            best_key = (float("inf"), float("inf"))

            for idx in range(n):
                x[idx] = 1 - x[idx]
                violation = self._violation(ir_model, row_terms, x)
                score_obj = self._objective(ir_model, x)
                tie_break = score_obj if ir_model.objective.sense == ObjectiveSense.MIN else -score_obj
                key = (violation, tie_break)
                if key < best_key:
                    best_key = key
                    best_idx = idx
                x[idx] = 1 - x[idx]

            if best_idx < 0 or best_key[0] >= current_violation - 1e-12:
                break

            x[best_idx] = 1 - x[best_idx]

        return x if self._is_feasible(ir_model, row_terms, x) else None

    def _local_search(
        self,
        ir_model: IRModel,
        row_terms: list[list[tuple[int, float]]],
        x: list[int],
        rng: random.Random,
        deadline: float | None,
    ) -> list[int]:
        indices = list(range(len(x)))
        while True:
            if deadline is not None and time.perf_counter() >= deadline:
                break

            improved = False
            incumbent = self._objective(ir_model, x)
            rng.shuffle(indices)

            for idx in indices:
                x[idx] = 1 - x[idx]
                if self._is_feasible(ir_model, row_terms, x):
                    candidate = self._objective(ir_model, x)
                    if self._is_better(ir_model, candidate, incumbent):
                        incumbent = candidate
                        improved = True
                        continue
                x[idx] = 1 - x[idx]

            if not improved:
                break

        return x

    def _constraint_pressure(self, ir_model: IRModel, row_terms: list[list[tuple[int, float]]]) -> list[float]:
        n = len(ir_model.variables)
        pressure = [0.0] * n
        if ir_model.constraints is None:
            return pressure

        for row in range(ir_model.constraints.num_rows):
            if ir_model.constraints.senses[row] != ConstraintSense.LE:
                continue
            rhs = max(1e-9, float(ir_model.constraints.rhs[row]))
            for col, coef in row_terms[row]:
                if coef > 0:
                    pressure[col] += coef / rhs
        return pressure

    def _row_terms(self, ir_model: IRModel) -> list[list[tuple[int, float]]]:
        if ir_model.constraints is None:
            return []

        rows = [[] for _ in range(ir_model.constraints.num_rows)]
        for row, col, value in zip(
            ir_model.constraints.row_indices,
            ir_model.constraints.col_indices,
            ir_model.constraints.values,
            strict=True,
        ):
            rows[row].append((col, float(value)))
        return rows

    def _respects_upper_bounds(self, ir_model: IRModel, row_terms: list[list[tuple[int, float]]], x: list[int]) -> bool:
        if ir_model.constraints is None:
            return True

        tol = 1e-9
        for row in range(ir_model.constraints.num_rows):
            if ir_model.constraints.senses[row] != ConstraintSense.LE:
                continue
            lhs = sum(coef * x[col] for col, coef in row_terms[row])
            if lhs > float(ir_model.constraints.rhs[row]) + tol:
                return False
        return True

    def _is_feasible(self, ir_model: IRModel, row_terms: list[list[tuple[int, float]]], x: list[int]) -> bool:
        if ir_model.constraints is None:
            return True

        tol = 1e-9
        for row in range(ir_model.constraints.num_rows):
            lhs = sum(coef * x[col] for col, coef in row_terms[row])
            rhs = float(ir_model.constraints.rhs[row])
            sense = ir_model.constraints.senses[row]

            if sense == ConstraintSense.LE and lhs > rhs + tol:
                return False
            if sense == ConstraintSense.GE and lhs < rhs - tol:
                return False
            if sense == ConstraintSense.EQ and abs(lhs - rhs) > tol:
                return False

        return True

    def _violation(self, ir_model: IRModel, row_terms: list[list[tuple[int, float]]], x: list[int]) -> float:
        if ir_model.constraints is None:
            return 0.0

        violation = 0.0
        for row in range(ir_model.constraints.num_rows):
            lhs = sum(coef * x[col] for col, coef in row_terms[row])
            rhs = float(ir_model.constraints.rhs[row])
            sense = ir_model.constraints.senses[row]

            if sense == ConstraintSense.LE:
                violation += max(0.0, lhs - rhs)
            elif sense == ConstraintSense.GE:
                violation += max(0.0, rhs - lhs)
            else:
                violation += abs(lhs - rhs)

        return violation

    def _objective(self, ir_model: IRModel, x: list[int]) -> float:
        value = float(ir_model.objective.constant)
        value += sum(float(c) * x[idx] for idx, c in enumerate(ir_model.objective.linear))
        return value

    def _is_better(self, ir_model: IRModel, candidate: float, incumbent: float | None) -> bool:
        if incumbent is None:
            return True
        if ir_model.objective.sense == ObjectiveSense.MAX:
            return candidate > incumbent + 1e-12
        return candidate < incumbent - 1e-12

    def _to_named_solution(self, ir_model: IRModel, x: list[int]) -> dict[str, float]:
        return {ir_model.variables[idx].name: float(x[idx]) for idx in range(len(x))}


def get_plugin() -> GraspBinaryProSolver:
    return GraspBinaryProSolver()
