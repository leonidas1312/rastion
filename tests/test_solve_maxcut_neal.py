from pathlib import Path

import pytest

from rastion.backends.local import LocalBackend
from rastion.compile.normalize import compile_to_ir
from rastion.core.data import InstanceData
from rastion.core.solution import SolutionStatus
from rastion.core.spec import ProblemSpec
from rastion.solvers.discovery import discover_plugins


@pytest.mark.integration
def test_solve_maxcut_with_neal() -> None:
    plugins = discover_plugins()
    if "neal" not in plugins:
        pytest.skip("neal plugin is not installed")

    spec = ProblemSpec.from_json_file(Path("examples/maxcut/spec.json"))
    instance = InstanceData.from_json_file(Path("examples/maxcut/instance.json"))
    ir_model = compile_to_ir(spec, instance)

    backend = LocalBackend()
    solution = backend.run(plugins["neal"], ir_model, {"num_reads": 50, "sweeps": 500, "seed": 0})

    assert solution.status in {SolutionStatus.OPTIMAL, SolutionStatus.FEASIBLE}
    assert len(solution.primal_values) == len(ir_model.variables)
