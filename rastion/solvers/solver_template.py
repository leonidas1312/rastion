"""Solver template - copy this file to create a custom solver.

Save as: solver.py in your GitHub repo.
"""

from __future__ import annotations

from rastion.core.ir import IRModel
from rastion.core.solution import Solution, SolutionStatus
from rastion.core.spec import VariableType
from rastion.solvers.base import (
    CapabilitySet,
    ConstraintType,
    ObjectiveType,
    SolveMode,
    SolverPlugin,
)


class MySolverPlugin(SolverPlugin):
    """Your solver name.

    Edit this class to implement your solver.
    """

    name = "my_solver"  # CHANGE THIS - unique name
    version = "0.1.0"  # CHANGE THIS - your version

    def capabilities(self) -> CapabilitySet:
        """Declare what your solver supports."""
        return CapabilitySet(
            variable_types={VariableType.BINARY},
            objective_types={ObjectiveType.LINEAR},
            constraint_types={ConstraintType.LINEAR},
            modes={SolveMode.SOLVE},
        )

    def solve(self, ir_model: IRModel, config: dict[str, object], backend: object) -> Solution:
        """Solve the optimization problem.

        ir_model contains:
        - ir_model.variables: list of variables
        - ir_model.objective: linear + quadratic objective
        - ir_model.constraints: linear constraints (if any)
        - ir_model.qubo: QUBO form (if ir_target is qubo)

        config contains solver parameters:
        - config.get("time_limit")
        - config.get("seed")
        - etc.

        Return Solution with:
        - status: OPTIMAL, FEASIBLE, INFEASIBLE, UNKNOWN, ERROR
        - objective_value: float
        - primal_values: dict of {variable_name: value}
        - metadata: dict with runtime_s, solver_name, solver_version
        """

        # TODO: Implement your solver here

        # Example: return dummy solution
        return Solution(
            status=SolutionStatus.FEASIBLE,
            objective_value=0.0,
            primal_values={v.name: 0.0 for v in ir_model.variables},
            metadata={
                "runtime_s": 0.0,
                "solver_name": self.name,
                "solver_version": self.version,
            },
        )


def get_plugin() -> MySolverPlugin | None:
    """Factory function - return None if solver dependencies not installed."""
    # TODO: Add dependency check if needed
    return MySolverPlugin()
