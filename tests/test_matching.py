from rastion.core.ir import QUBOIR, IRModel, IRObjective, IRVariable
from rastion.core.solution import Solution, SolutionStatus
from rastion.core.spec import IRTarget, ObjectiveSense, VariableType
from rastion.solvers.base import CapabilitySet, ConstraintType, ObjectiveType, SolveMode, SolverPlugin
from rastion.solvers.matching import auto_select_solver


class DummyPlugin(SolverPlugin):
    def __init__(self, name: str, capability_set: CapabilitySet) -> None:
        self.name = name
        self.version = "test"
        self._capability_set = capability_set

    def capabilities(self) -> CapabilitySet:
        return self._capability_set

    def solve(self, ir_model: IRModel, config: dict[str, object], backend: object) -> Solution:
        return Solution(status=SolutionStatus.UNKNOWN)


def test_auto_select_solver_matches_qubo_sampler() -> None:
    ir_model = IRModel(
        schema_version="0.1.0",
        target=IRTarget.QUBO,
        variables=[
            IRVariable(index=0, name="x0", vartype=VariableType.BINARY, lb=0, ub=1),
            IRVariable(index=1, name="x1", vartype=VariableType.BINARY, lb=0, ub=1),
        ],
        objective=IRObjective(sense=ObjectiveSense.MIN, linear=[0.0, 0.0]),
        constraints=None,
        qubo=QUBOIR(n=2, q_i=[0], q_j=[1], q_v=[1.0], linear=[0.0, 0.0], constant=0.0),
    )

    milp_plugin = DummyPlugin(
        "milp",
        CapabilitySet(
            variable_types={VariableType.BINARY, VariableType.INTEGER, VariableType.CONTINUOUS},
            objective_types={ObjectiveType.LINEAR},
            constraint_types={ConstraintType.LINEAR},
            modes={SolveMode.SOLVE},
        ),
    )
    qubo_plugin = DummyPlugin(
        "qubo",
        CapabilitySet(
            variable_types={VariableType.BINARY},
            objective_types={ObjectiveType.QUBO},
            constraint_types=set(),
            modes={SolveMode.SAMPLE},
        ),
    )

    selected = auto_select_solver(ir_model, {"milp": milp_plugin, "qubo": qubo_plugin})
    assert selected.name == "qubo"
