from __future__ import annotations

import random
import time

from rastion.core.ir import IRModel
from rastion.core.solution import Solution, SolutionStatus
from rastion.core.spec import ObjectiveSense, VariableType
from rastion.solvers.base import CapabilitySet, ObjectiveType, SolveMode, SolverPlugin


class QuboTabuProSolver(SolverPlugin):
    """Tabu-search sampler for QUBO models."""

    name = "qubo_tabu_pro"
    version = "0.1.0"

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            variable_types={VariableType.BINARY},
            objective_types={ObjectiveType.QUBO},
            constraint_types=set(),
            modes={SolveMode.SAMPLE, SolveMode.SOLVE},
            result_fields={"objective_value", "primal_values", "runtime_s"},
        )

    def solve(self, ir_model: IRModel, config: dict[str, object], backend: object) -> Solution:
        if ir_model.qubo is None:
            raise ValueError("qubo_tabu_pro requires QUBO IR")
        if ir_model.constraints is not None and ir_model.constraints.num_rows > 0:
            raise ValueError("qubo_tabu_pro does not support linear constraints")
        if any(var.vartype != VariableType.BINARY for var in ir_model.variables):
            raise ValueError("qubo_tabu_pro supports binary variables only")

        start = time.perf_counter()
        n = ir_model.qubo.n
        rng = random.Random(int(config.get("seed", 0)))

        time_limit = float(config.get("time_limit", 0.0)) if "time_limit" in config else None
        deadline = (start + time_limit) if time_limit and time_limit > 0 else None

        iterations = int(config.get("iterations", max(3000, 80 * n)))
        restarts = int(config.get("restarts", max(4, n // 20)))
        tenure = max(5, int(config.get("tabu_tenure", max(8, n // 6))))
        stagnation_limit = int(config.get("stagnation_limit", max(250, 5 * n)))

        linear, neighbors = self._build_qubo(ir_model)
        minimize = ir_model.objective.sense == ObjectiveSense.MIN

        best_x: list[int] | None = None
        best_adjusted: float | None = None
        best_raw = 0.0

        seed_vectors = self._seed_vectors(n, restarts, rng)
        for restart_id, x in enumerate(seed_vectors):
            if deadline is not None and time.perf_counter() >= deadline:
                break

            fields = self._fields(linear, neighbors, x)
            raw = self._raw_energy(ir_model, linear, neighbors, x)
            adjusted = raw if minimize else -raw

            tabu_until = [0] * n
            stagnation = 0

            if best_adjusted is None or adjusted < best_adjusted:
                best_x = list(x)
                best_adjusted = adjusted
                best_raw = raw

            for step in range(max(1, iterations)):
                if deadline is not None and time.perf_counter() >= deadline:
                    break

                chosen_idx = -1
                chosen_delta_raw = 0.0
                chosen_delta_adjusted = 0.0
                best_move = float("inf")

                for idx in range(n):
                    delta_raw = (1 - 2 * x[idx]) * fields[idx]
                    delta_adjusted = delta_raw if minimize else -delta_raw
                    candidate_adjusted = adjusted + delta_adjusted

                    tabu_active = step < tabu_until[idx]
                    aspiration = best_adjusted is None or candidate_adjusted < best_adjusted - 1e-12
                    if tabu_active and not aspiration:
                        continue

                    if delta_adjusted < best_move - 1e-12:
                        best_move = delta_adjusted
                        chosen_idx = idx
                        chosen_delta_raw = delta_raw
                        chosen_delta_adjusted = delta_adjusted

                if chosen_idx < 0:
                    chosen_idx = rng.randrange(n)
                    chosen_delta_raw = (1 - 2 * x[chosen_idx]) * fields[chosen_idx]
                    chosen_delta_adjusted = chosen_delta_raw if minimize else -chosen_delta_raw

                old_value = x[chosen_idx]
                x[chosen_idx] = 1 - x[chosen_idx]
                change = 1 - 2 * old_value

                for neighbor, weight in neighbors[chosen_idx]:
                    fields[neighbor] += weight * change

                raw += chosen_delta_raw
                adjusted += chosen_delta_adjusted

                tabu_until[chosen_idx] = step + tenure + rng.randint(0, max(1, tenure // 2))

                if best_adjusted is None or adjusted < best_adjusted - 1e-12:
                    best_adjusted = adjusted
                    best_raw = raw
                    best_x = list(x)
                    stagnation = 0
                else:
                    stagnation += 1

                if stagnation >= stagnation_limit:
                    break

            if restart_id + 1 >= restarts:
                break

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
                    "method": "tabu",
                },
            )

        objective_value = best_raw + float(ir_model.qubo.constant)
        return Solution(
            status=SolutionStatus.FEASIBLE,
            objective_value=objective_value,
            primal_values={ir_model.variables[i].name: float(best_x[i]) for i in range(n)},
            metadata={
                "runtime_s": runtime_s,
                "solver_name": self.name,
                "solver_version": self.version,
                "method": "tabu",
                "iterations": iterations,
                "restarts": restarts,
                "tabu_tenure": tenure,
            },
        )

    def _build_qubo(self, ir_model: IRModel) -> tuple[list[float], list[list[tuple[int, float]]]]:
        assert ir_model.qubo is not None

        n = ir_model.qubo.n
        linear = [float(v) for v in ir_model.qubo.linear]
        weights: dict[tuple[int, int], float] = {}

        for i_raw, j_raw, value_raw in zip(ir_model.qubo.q_i, ir_model.qubo.q_j, ir_model.qubo.q_v, strict=True):
            i = int(i_raw)
            j = int(j_raw)
            value = float(value_raw)

            if i == j:
                linear[i] += value
                continue

            a, b = (i, j) if i < j else (j, i)
            weights[(a, b)] = weights.get((a, b), 0.0) + value

        neighbors = [[] for _ in range(n)]
        for (i, j), weight in weights.items():
            neighbors[i].append((j, weight))
            neighbors[j].append((i, weight))

        return linear, neighbors

    def _fields(self, linear: list[float], neighbors: list[list[tuple[int, float]]], x: list[int]) -> list[float]:
        fields = [0.0] * len(x)
        for i in range(len(x)):
            value = linear[i]
            for j, weight in neighbors[i]:
                value += weight * x[j]
            fields[i] = value
        return fields

    def _raw_energy(
        self,
        ir_model: IRModel,
        linear: list[float],
        neighbors: list[list[tuple[int, float]]],
        x: list[int],
    ) -> float:
        value = 0.0
        for i in range(len(x)):
            value += linear[i] * x[i]
            for j, weight in neighbors[i]:
                if i < j:
                    value += weight * x[i] * x[j]
        return value

    def _seed_vectors(self, n: int, restarts: int, rng: random.Random) -> list[list[int]]:
        seeds = [[0] * n]
        if n > 0:
            seeds.append([1] * n)
        for _ in range(max(0, restarts - len(seeds))):
            seeds.append([rng.randint(0, 1) for _ in range(n)])
        return seeds


def get_plugin() -> QuboTabuProSolver:
    return QuboTabuProSolver()
