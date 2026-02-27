"""Backend protocol for executing solver plugins."""

from __future__ import annotations

from typing import Protocol

from rastion.core.ir import IRModel
from rastion.core.solution import Solution


class Backend(Protocol):
    def run(self, plugin: object, ir_model: IRModel, config: dict[str, object]) -> Solution:
        ...
