from pathlib import Path

import numpy as np

from rastion.core.data import InstanceData
from rastion.core.spec import ProblemSpec


def test_problem_spec_round_trip(tmp_path: Path) -> None:
    src = Path("examples/knapsack/spec.json")
    spec = ProblemSpec.from_json_file(src)

    out = tmp_path / "spec_round_trip.json"
    spec.to_json_file(out)
    loaded = ProblemSpec.from_json_file(out)

    assert loaded.model_dump(mode="json") == spec.model_dump(mode="json")


def test_instance_round_trip_with_npz_payload(tmp_path: Path) -> None:
    instance = InstanceData.model_validate(
        {
            "schema_version": "0.1.0",
            "arrays": {
                "c": [1.0, 2.0, 3.0],
                "A": [[1.0, 0.0, 2.0]],
            },
            "params": {"name": "demo"},
        }
    )

    out_json = tmp_path / "instance.json"
    instance.to_json_file(out_json, npz_payload="payload.npz")
    loaded = InstanceData.from_json_file(out_json)

    np.testing.assert_allclose(loaded.get_array("c"), np.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(loaded.get_array("A"), np.asarray([[1.0, 0.0, 2.0]]))
    assert loaded.params["name"] == "demo"
