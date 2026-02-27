"""Local in-process backend."""

from __future__ import annotations

from rastion.core.ir import IRModel
from rastion.core.solution import Solution


class LocalBackend:
    """Executes solver plugin code in-process."""

    name = "local"

    def run(self, plugin: object, ir_model: IRModel, config: dict[str, object]) -> Solution:
        solve = getattr(plugin, "solve", None)
        if solve is None:
            raise TypeError("solver plugin is missing solve()")
        return solve(ir_model=ir_model, config=config, backend=self)
