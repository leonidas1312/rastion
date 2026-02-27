"""Run record model and persistence utilities."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from rastion.core.data import InstanceData
from rastion.core.solution import Solution
from rastion.core.spec import ProblemSpec
from rastion.version import __version__


class RunRecord(BaseModel):
    timestamp: str
    rastion_version: str
    python_version: str
    solver_name: str
    solver_version: str
    solver_config: dict[str, Any] = Field(default_factory=dict)
    problem_hash: str
    solution: Solution


def compute_problem_hash(spec: ProblemSpec, instance: InstanceData) -> str:
    payload = {
        "spec": spec.model_dump(mode="json"),
        "instance": instance.to_serializable_dict(),
    }
    normalized = _normalize(payload)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_run_record(
    spec: ProblemSpec,
    instance: InstanceData,
    solution: Solution,
    solver_name: str,
    solver_version: str,
    solver_config: dict[str, Any],
) -> RunRecord:
    return RunRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        rastion_version=__version__,
        python_version=platform.python_version(),
        solver_name=solver_name,
        solver_version=solver_version,
        solver_config=solver_config,
        problem_hash=compute_problem_hash(spec, instance),
        solution=solution,
    )


def append_run_record(record: RunRecord, runs_dir: str | Path = "runs") -> Path:
    out_dir = Path(runs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "runs.jsonl"
    with out_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.model_dump(mode="json"), sort_keys=True))
        f.write("\n")
    return out_file


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, tuple):
        return [_normalize(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
