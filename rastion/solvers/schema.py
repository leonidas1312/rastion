from __future__ import annotations

from pydantic import BaseModel, Field


class SolverMetadata(BaseModel):
    name: str
    supports: list[str] = Field(default_factory=list)
    max_size: int = Field(gt=0)
    hardware: list[str] = Field(default_factory=lambda: ["cpu"])
    quality: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = "unknown"
    version: str | None = None
