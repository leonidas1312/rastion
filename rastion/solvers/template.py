"""Custom solver integration template.

This file shows how to add a new solver plugin to rastion.
Copy this file, rename the class, and implement the interface.
"""

from __future__ import annotations

from rastion.core.ir import IRModel
from rastion.core.solution import Solution
from rastion.core.spec import VariableType
from rastion.solvers.base import CapabilitySet, ConstraintType, ObjectiveType, SolveMode, SolverPlugin


class MySolverPlugin(SolverPlugin):
    """Template solver plugin.

    Replace this class with your solver-specific implementation.
    """

    # Unique plugin identifier used by CLI and discovery.
    name = "my_solver"

    # Version of the wrapped solver library.
    version = "1.0.0"

    def capabilities(self) -> CapabilitySet:
        """Declare exactly what your solver supports.

        Update this declaration to match your implementation:
        - variable_types: {BINARY, INTEGER, CONTINUOUS}
        - objective_types: {LINEAR, QUADRATIC, QUBO}
        - constraint_types: {LINEAR}
        - modes: {SOLVE, SAMPLE}
        - supports_miqp: bool
        """

        return CapabilitySet(
            variable_types={VariableType.BINARY},
            objective_types={ObjectiveType.LINEAR},
            constraint_types={ConstraintType.LINEAR},
            modes={SolveMode.SOLVE},
            supports_miqp=False,
            result_fields={"objective_value", "primal_values", "runtime_s"},
        )

    def solve(self, ir_model: IRModel, config: dict[str, object], backend: object) -> Solution:
        """Solve a problem in four steps.

        1) Validate input and raise ValueError for unsupported features.
        2) Convert IRModel into your solver-native model.
        3) Call the solver with config options.
        4) Convert the solver result into Solution.
        """

        # Replace with your real conversion + solve logic.
        raise NotImplementedError("Implement solver conversion and execution")


def get_plugin() -> MySolverPlugin | None:
    """Factory used by plugin discovery.

    Return None if the underlying solver package is not installed.
    """

    try:
        import my_solver_library  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return None

    # Use the real version from your solver package when available.
    return MySolverPlugin()
