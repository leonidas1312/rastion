"""Built-in baseline solver plugin for local-first fallback."""

from __future__ import annotations

import random
import time

from rastion.core.ir import IRModel
from rastion.core.solution import Solution, SolutionStatus
from rastion.core.spec import ConstraintSense, ObjectiveSense, VariableType
from rastion.solvers.base import CapabilitySet, ConstraintType, ObjectiveType, SolveMode, SolverPlugin


class BaselinePlugin(SolverPlugin):
    """Simple binary linear solver: brute-force for small n, heuristic otherwise."""

    name = "baseline"
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
        if any(var.vartype != VariableType.BINARY for var in ir_model.variables):
            raise ValueError("baseline solver supports only binary variables")
        if ir_model.qubo is not None:
            raise ValueError("baseline solver does not support QUBO target")
        if ir_model.objective.q_v:
            raise ValueError("baseline solver does not support quadratic objectives")

        start = time.perf_counter()
        time_limit = float(config.get("time_limit", 0.0)) if "time_limit" in config else None
        deadline = (start + time_limit) if time_limit and time_limit > 0 else None

        n = len(ir_model.variables)
        row_terms = self._row_terms(ir_model)

        if n <= 25:
            best_x, best_obj, completed = self._solve_bruteforce(ir_model, row_terms, deadline)
            runtime_s = time.perf_counter() - start

            if best_x is None:
                status = SolutionStatus.INFEASIBLE if completed else SolutionStatus.UNKNOWN
                return Solution(
                    status=status,
                    objective_value=None,
                    primal_values={},
                    metadata={
                        "runtime_s": runtime_s,
                        "solver_name": self.name,
                        "solver_version": self.version,
                        "method": "bruteforce",
                    },
                )

            status = SolutionStatus.OPTIMAL if completed else SolutionStatus.FEASIBLE
            return Solution(
                status=status,
                objective_value=best_obj,
                primal_values=self._to_named_solution(ir_model, best_x),
                metadata={
                    "runtime_s": runtime_s,
                    "solver_name": self.name,
                    "solver_version": self.version,
                    "method": "bruteforce",
                },
            )

        best_x, best_obj = self._solve_heuristic(ir_model, row_terms, deadline, config)
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
                    "method": "heuristic",
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
                "method": "heuristic",
            },
        )

    def _solve_bruteforce(
        self,
        ir_model: IRModel,
        row_terms: list[list[tuple[int, float]]],
        deadline: float | None,
    ) -> tuple[list[int] | None, float | None, bool]:
        n = len(ir_model.variables)
        total = 1 << n

        best_x: list[int] | None = None
        best_obj: float | None = None
        completed = True

        for mask in range(total):
            if deadline is not None and mask % 2048 == 0 and time.perf_counter() >= deadline:
                completed = False
                break

            x = [(mask >> i) & 1 for i in range(n)]
            if not self._is_feasible(ir_model, row_terms, x):
                continue
            value = self._objective(ir_model, x)
            if self._is_better(ir_model, value, best_obj):
                best_obj = value
                best_x = x

        return best_x, best_obj, completed

    def _solve_heuristic(
        self,
        ir_model: IRModel,
        row_terms: list[list[tuple[int, float]]],
        deadline: float | None,
        config: dict[str, object],
    ) -> tuple[list[int] | None, float | None]:
        n = len(ir_model.variables)
        rng = random.Random(int(config.get("seed", 0)))

        best_x: list[int] | None = None
        best_obj: float | None = None

        seeds = [
            [0] * n,
            self._greedy_start(ir_model, row_terms),
        ]

        random_starts = int(config.get("random_starts", 12))
        for _ in range(random_starts):
            seeds.append([rng.randint(0, 1) for _ in range(n)])

        for seed in seeds:
            if deadline is not None and time.perf_counter() >= deadline:
                break

            x = list(seed)
            if not self._is_feasible(ir_model, row_terms, x):
                x = self._repair_to_feasible(ir_model, row_terms, x, deadline)
            if x is None:
                continue

            x = self._hillclimb(ir_model, row_terms, x, deadline)
            value = self._objective(ir_model, x)
            if self._is_better(ir_model, value, best_obj):
                best_x = list(x)
                best_obj = value

        return best_x, best_obj

    def _greedy_start(self, ir_model: IRModel, row_terms: list[list[tuple[int, float]]]) -> list[int]:
        n = len(ir_model.variables)
        x = [0] * n

        objective_pairs = list(enumerate(ir_model.objective.linear))
        if ir_model.objective.sense == ObjectiveSense.MAX:
            objective_pairs.sort(key=lambda item: item[1], reverse=True)
        else:
            objective_pairs.sort(key=lambda item: item[1])

        for idx, coeff in objective_pairs:
            if ir_model.objective.sense == ObjectiveSense.MAX and coeff <= 0:
                continue
            if ir_model.objective.sense == ObjectiveSense.MIN and coeff >= 0:
                continue
            x[idx] = 1
            if not self._is_feasible(ir_model, row_terms, x):
                x[idx] = 0

        return x

    def _repair_to_feasible(
        self,
        ir_model: IRModel,
        row_terms: list[list[tuple[int, float]]],
        x: list[int],
        deadline: float | None,
    ) -> list[int] | None:
        for _ in range(max(8, 4 * len(x))):
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            if self._is_feasible(ir_model, row_terms, x):
                return x

            best_idx = -1
            best_violation = self._violation(ir_model, row_terms, x)
            for idx in range(len(x)):
                x[idx] = 1 - x[idx]
                violation = self._violation(ir_model, row_terms, x)
                if violation + 1e-12 < best_violation:
                    best_violation = violation
                    best_idx = idx
                x[idx] = 1 - x[idx]

            if best_idx < 0:
                break
            x[best_idx] = 1 - x[best_idx]

        return x if self._is_feasible(ir_model, row_terms, x) else None

    def _hillclimb(
        self,
        ir_model: IRModel,
        row_terms: list[list[tuple[int, float]]],
        x: list[int],
        deadline: float | None,
    ) -> list[int]:
        improved = True
        while improved:
            if deadline is not None and time.perf_counter() >= deadline:
                break

            improved = False
            incumbent = self._objective(ir_model, x)
            best_idx = -1
            best_value = incumbent

            for idx in range(len(x)):
                x[idx] = 1 - x[idx]
                if self._is_feasible(ir_model, row_terms, x):
                    candidate = self._objective(ir_model, x)
                    if self._is_better(ir_model, candidate, best_value):
                        best_value = candidate
                        best_idx = idx
                x[idx] = 1 - x[idx]

            if best_idx >= 0:
                x[best_idx] = 1 - x[best_idx]
                improved = True

        return x

    def _row_terms(self, ir_model: IRModel) -> list[list[tuple[int, float]]]:
        if ir_model.constraints is None:
            return []
        row_terms: list[list[tuple[int, float]]] = [
            [] for _ in range(ir_model.constraints.num_rows)
        ]
        for row, col, value in zip(
            ir_model.constraints.row_indices,
            ir_model.constraints.col_indices,
            ir_model.constraints.values,
            strict=True,
        ):
            row_terms[row].append((col, float(value)))
        return row_terms

    def _is_feasible(self, ir_model: IRModel, row_terms: list[list[tuple[int, float]]], x: list[int]) -> bool:
        if ir_model.constraints is None:
            return True

        tol = 1e-9
        for row in range(ir_model.constraints.num_rows):
            lhs = sum(coef * x[col] for col, coef in row_terms[row])
            rhs = ir_model.constraints.rhs[row]
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

        total = 0.0
        for row in range(ir_model.constraints.num_rows):
            lhs = sum(coef * x[col] for col, coef in row_terms[row])
            rhs = ir_model.constraints.rhs[row]
            sense = ir_model.constraints.senses[row]

            if sense == ConstraintSense.LE:
                total += max(0.0, lhs - rhs)
            elif sense == ConstraintSense.GE:
                total += max(0.0, rhs - lhs)
            else:
                total += abs(lhs - rhs)
        return total

    def _objective(self, ir_model: IRModel, x: list[int]) -> float:
        value = float(ir_model.objective.constant)
        value += sum(c * x[i] for i, c in enumerate(ir_model.objective.linear))
        return float(value)

    def _is_better(self, ir_model: IRModel, candidate: float, incumbent: float | None) -> bool:
        if incumbent is None:
            return True
        if ir_model.objective.sense == ObjectiveSense.MAX:
            return candidate > incumbent + 1e-12
        return candidate < incumbent - 1e-12

    def _to_named_solution(self, ir_model: IRModel, x: list[int]) -> dict[str, float]:
        return {ir_model.variables[i].name: float(x[i]) for i in range(len(x))}


def get_plugin() -> BaselinePlugin:
    return BaselinePlugin()
