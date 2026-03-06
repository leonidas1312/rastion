from __future__ import annotations

from typing import Any


def detect_problem_type(raw: dict[str, Any]) -> str:
    raw_type = raw.get("type") or raw.get("problem_type")
    if not isinstance(raw_type, str):
        raise ValueError("Problem JSON must include a string 'type'")
    problem_type = raw_type.lower().strip()
    if problem_type != "qubo":
        raise ValueError(f"Unsupported problem type for MVP: {raw_type}")
    return problem_type
