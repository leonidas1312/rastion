from rastion.core.data import InstanceData
from rastion.core.spec import ProblemSpec
from rastion.core.validate import validate_problem_and_instance


def test_validation_fails_on_bad_objective_dimension() -> None:
    spec = ProblemSpec.model_validate(
        {
            "schema_version": "0.1.0",
            "name": "bad_dims",
            "variables": [
                {"name": "x0", "vartype": "binary"},
                {"name": "x1", "vartype": "binary"},
            ],
            "objective": {"sense": "max", "linear": "c"},
            "constraints": [],
        }
    )
    instance = InstanceData.model_validate(
        {
            "schema_version": "0.1.0",
            "arrays": {
                "c": [1.0, 2.0, 3.0],
            },
        }
    )

    result = validate_problem_and_instance(spec, instance)
    assert not result.valid
    assert any("objective linear 'c' has length" in e for e in result.errors)
