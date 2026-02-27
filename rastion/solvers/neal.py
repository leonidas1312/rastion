"""dwave-neal QUBO sampler adapter."""

from __future__ import annotations

import time

from rastion.core.ir import IRModel
from rastion.core.solution import Solution, SolutionStatus
from rastion.core.spec import VariableType
from rastion.solvers.base import CapabilitySet, ObjectiveType, SolveMode, SolverPlugin


class NealPlugin(SolverPlugin):
    name = "neal"

    def __init__(self, version: str) -> None:
        self.version = version

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            variable_types={VariableType.BINARY},
            objective_types={ObjectiveType.QUBO},
            constraint_types=set(),
            modes={SolveMode.SAMPLE},
            result_fields={"objective_value", "primal_values", "runtime_s"},
        )

    def solve(self, ir_model: IRModel, config: dict[str, object], backend: object) -> Solution:
        if ir_model.qubo is None:
            raise ValueError("neal plugin requires QUBO IR")
        if ir_model.constraints is not None and ir_model.constraints.num_rows > 0:
            raise ValueError("neal plugin does not support linear constraints")

        import neal

        qubo_dict: dict[tuple[int, int], float] = {}
        for i, j, value in zip(ir_model.qubo.q_i, ir_model.qubo.q_j, ir_model.qubo.q_v, strict=True):
            qubo_dict[(i, j)] = qubo_dict.get((i, j), 0.0) + float(value)

        for idx, value in enumerate(ir_model.qubo.linear):
            if value:
                qubo_dict[(idx, idx)] = qubo_dict.get((idx, idx), 0.0) + float(value)

        num_reads = int(config.get("num_reads", 100))
        sweeps = int(config.get("sweeps", 1000))
        seed = config.get("seed")

        sampler = neal.SimulatedAnnealingSampler()

        kwargs = {"num_reads": num_reads, "num_sweeps": sweeps}
        if seed is not None:
            kwargs["seed"] = int(seed)

        start = time.perf_counter()
        sampleset = sampler.sample_qubo(qubo_dict, **kwargs)
        runtime_s = time.perf_counter() - start

        best = sampleset.first
        sample = dict(best.sample)
        primal_values = {
            ir_model.variables[idx].name: float(sample.get(idx, 0))
            for idx in range(len(ir_model.variables))
        }

        energy = float(best.energy) + float(ir_model.qubo.constant)

        return Solution(
            status=SolutionStatus.FEASIBLE,
            objective_value=energy,
            primal_values=primal_values,
            metadata={
                "runtime_s": runtime_s,
                "solver_name": self.name,
                "solver_version": self.version,
                "num_reads": num_reads,
                "sweeps": sweeps,
            },
        )


def get_plugin() -> NealPlugin | None:
    try:
        import neal
    except ImportError:
        return None
    version = getattr(neal, "__version__", "unknown")
    return NealPlugin(version=version)
