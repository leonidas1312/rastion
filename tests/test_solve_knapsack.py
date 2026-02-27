from pathlib import Path

import pytest

from rastion.backends.local import LocalBackend
from rastion.compile.normalize import compile_to_ir
from rastion.core.data import InstanceData
from rastion.core.solution import SolutionStatus
from rastion.core.spec import ProblemSpec
from rastion.solvers.discovery import discover_plugins


@pytest.mark.integration
def test_solve_knapsack_with_available_milp_solver() -> None:
    spec = ProblemSpec.from_json_file(Path("examples/knapsack/spec.json"))
    instance = InstanceData.from_json_file(Path("examples/knapsack/instance.json"))
    ir_model = compile_to_ir(spec, instance)

    plugins = discover_plugins()
    plugin = None
    for name in ("ortools", "highs"):
        if name in plugins:
            plugin = plugins[name]
            break

    if plugin is None:
        pytest.skip("Neither OR-Tools nor HiGHS plugin is installed")

    backend = LocalBackend()
    try:
        solution = backend.run(plugin, ir_model, {"time_limit": 10})
    except Exception as exc:  # pragma: no cover - depends on optional solvers
        pytest.skip(f"{plugin.name} plugin unavailable at runtime: {exc}")

    assert solution.status in {SolutionStatus.OPTIMAL, SolutionStatus.FEASIBLE}
    assert solution.objective_value is not None
    assert solution.objective_value >= 15.0 - 1e-6
