from pathlib import Path

import numpy as np

from rastion.ir.convert import load_problem_json


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ir_conversion_shapes_and_types() -> None:
    problem = load_problem_json(REPO_ROOT / "examples" / "qubo_small.json")

    assert problem.problem_type == "qubo"
    assert problem.n_vars == 4
    assert problem.objective.Q.shape == (4, 4)
    assert problem.objective.c.shape == (4,)
    assert isinstance(problem.objective.Q, np.ndarray)
    assert isinstance(problem.objective.c, np.ndarray)
    assert all(v.domain == "binary" for v in problem.variables)
