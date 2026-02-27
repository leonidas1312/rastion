"""Instance data models and serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from rastion.core.spec import CURRENT_SCHEMA_VERSION, ensure_schema_compatible


class InstanceData(BaseModel):
    schema_version: str = CURRENT_SCHEMA_VERSION
    arrays: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    npz_payload: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "InstanceData":
        source = Path(path)
        with source.open("r", encoding="utf-8") as f:
            data = json.load(f)
        model = cls.model_validate(data)
        ensure_schema_compatible(model.schema_version, "InstanceData")

        merged = dict(model.arrays)
        if model.npz_payload:
            npz_path = (source.parent / model.npz_payload).resolve()
            loaded = _load_npz(npz_path)
            loaded.update(merged)
            merged = loaded

        return cls(
            schema_version=model.schema_version,
            arrays=merged,
            params=model.params,
            npz_payload=model.npz_payload,
        )

    @classmethod
    def from_npz_file(cls, path: str | Path) -> "InstanceData":
        arrays = _load_npz(Path(path))
        return cls(schema_version=CURRENT_SCHEMA_VERSION, arrays=arrays)

    @classmethod
    def from_path(cls, path: str | Path) -> "InstanceData":
        source = Path(path)
        if source.suffix.lower() == ".json":
            return cls.from_json_file(source)
        if source.suffix.lower() == ".npz":
            return cls.from_npz_file(source)
        raise ValueError(f"unsupported instance file extension: {source.suffix}")

    def get_array(self, key: str, ndim: int | None = None) -> np.ndarray:
        if key not in self.arrays:
            raise KeyError(f"instance array '{key}' is missing")
        value = self.arrays[key]
        arr = np.asarray(value)
        if ndim is not None and arr.ndim != ndim:
            raise ValueError(f"array '{key}' expected ndim={ndim}, found ndim={arr.ndim}")
        return arr

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "arrays": {k: _to_jsonable(v) for k, v in sorted(self.arrays.items())},
            "params": self.params,
        }

    def to_json_file(self, path: str | Path, npz_payload: str | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "params": self.params,
        }

        if npz_payload:
            numeric: dict[str, np.ndarray] = {}
            inline: dict[str, Any] = {}
            for key, value in self.arrays.items():
                arr = np.asarray(value)
                if arr.dtype.kind in {"b", "i", "u", "f"}:
                    numeric[key] = arr
                else:
                    inline[key] = _to_jsonable(value)
            npz_path = target.parent / npz_payload
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz_path, **numeric)
            payload["arrays"] = inline
            payload["npz_payload"] = npz_payload
        else:
            payload["arrays"] = {k: _to_jsonable(v) for k, v in sorted(self.arrays.items())}

        with target.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def to_npz_file(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        numeric: dict[str, np.ndarray] = {}
        for key, value in self.arrays.items():
            arr = np.asarray(value)
            if arr.dtype.kind not in {"b", "i", "u", "f"}:
                raise ValueError(f"array '{key}' is not numeric and cannot be saved to NPZ")
            numeric[key] = arr
        np.savez_compressed(target, **numeric)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value
