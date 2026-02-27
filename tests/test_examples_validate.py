from pathlib import Path

from rastion.compile.normalize import compile_to_ir
from rastion.core.data import InstanceData
from rastion.core.spec import ProblemSpec
from rastion.core.validate import validate_problem_and_instance


def test_new_examples_validate_and_compile() -> None:
    example_roots = [
        Path("examples/portfolio"),
        Path("examples/facility_location"),
        Path("examples/tsp"),
        Path("examples/set_cover"),
    ]

    for root in example_roots:
        spec = ProblemSpec.from_json_file(root / "spec.json")
        instance = InstanceData.from_json_file(root / "instance.json")
        result = validate_problem_and_instance(spec, instance)
        assert result.valid, f"validation failed for {root}: {result.errors}"
        _ = compile_to_ir(spec, instance)
