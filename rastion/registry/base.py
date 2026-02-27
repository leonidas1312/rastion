"""Registry entry models for local and future remote registries."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

_SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)


class RegistryEntry(BaseModel):
    """Base metadata shared by all registry entries."""

    name: str
    version: str = "0.1.0"
    author: str = "unknown"
    tags: list[str] = Field(default_factory=list)
    description: str = ""

    @model_validator(mode="after")
    def _validate_version(self) -> "RegistryEntry":
        if not _SEMVER_RE.match(self.version):
            raise ValueError(f"version must be semantic (got: {self.version!r})")
        return self


class ProblemEntry(RegistryEntry):
    """Problem metadata stored in the local registry."""

    optimization_class: str = "unknown"
    difficulty: str = "medium"
    path: Path | None = None


class SolverEntry(RegistryEntry):
    """Solver metadata exposed by the registry manager."""

    open_source: bool = True
    capabilities: dict[str, object] = Field(default_factory=dict)
