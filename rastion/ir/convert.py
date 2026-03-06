from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .detect import detect_problem_type
from .types import Objective, ProblemIR, Variable


def normalize_problem_dict(raw: dict[str, Any]) -> ProblemIR:
    problem_type = detect_problem_type(raw)
    q = np.asarray(raw.get("Q"), dtype=float)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("Q must be a square matrix")

    n_vars = q.shape[0]
    c_raw = raw.get("c")
    if c_raw is None:
        c = np.zeros(n_vars, dtype=float)
    else:
        c = np.asarray(c_raw, dtype=float)

    variables = [Variable(name=f"x{i}", domain="binary") for i in range(n_vars)]
    objective = Objective(
        Q=q,
        c=c,
        constant=float(raw.get("constant", 0.0)),
    )

    return ProblemIR(
        name=str(raw.get("name", "unnamed_problem")),
        problem_type=problem_type,
        n_vars=n_vars,
        variables=variables,
        objective=objective,
        constraints=raw.get("constraints"),
    )


def load_problem_json(path: str | Path) -> ProblemIR:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Problem file must contain a JSON object")
    return normalize_problem_dict(data)
