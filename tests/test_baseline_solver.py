from pathlib import Path

from rastion.backends.local import LocalBackend
from rastion.compile.normalize import compile_to_ir
from rastion.core.data import InstanceData
from rastion.core.solution import SolutionStatus
from rastion.core.spec import ProblemSpec
from rastion.solvers.discovery import discover_plugins


def test_baseline_solver_solves_knapsack_to_optimal_value() -> None:
    spec = ProblemSpec.from_json_file(Path("examples/knapsack/spec.json"))
    instance = InstanceData.from_json_file(Path("examples/knapsack/instance.json"))
    ir_model = compile_to_ir(spec, instance)

    plugins = discover_plugins()
    assert "baseline" in plugins

    solution = LocalBackend().run(plugins["baseline"], ir_model, {"time_limit": 5})

    assert solution.status in {SolutionStatus.OPTIMAL, SolutionStatus.FEASIBLE}
    assert solution.objective_value is not None
    assert solution.objective_value == 15.0
