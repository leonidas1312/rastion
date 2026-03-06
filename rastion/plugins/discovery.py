from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from importlib.metadata import entry_points
from pathlib import Path
from types import ModuleType

from rastion.solvers.base import Solver
from rastion.solvers.schema import SolverMetadata
from rastion.solvers.registry import clear_registry, register_solver


LOCAL_PLUGINS_DIR = Path("plugins_local")
ENTRYPOINT_GROUP = "rastion.solvers"


def _load_module_from_path(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _metadata_from_solver(solver: Solver, source: str, quality: float | None = None) -> SolverMetadata:
    return SolverMetadata(
        name=solver.name,
        supports=list(solver.supports),
        max_size=int(solver.max_size),
        hardware=list(solver.hardware),
        quality=solver.quality if quality is None else quality,
        source=source,
    )


def _load_local_solvers(plugin_root: Path) -> list[tuple[Solver, SolverMetadata]]:
    discovered: list[tuple[Solver, SolverMetadata]] = []
    if not plugin_root.exists():
        return discovered

    for plugin_dir in sorted(p for p in plugin_root.iterdir() if p.is_dir()):
        metadata_path = plugin_dir / plugin_dir.name / "metadata.json"
        solver_path = plugin_dir / plugin_dir.name / "solver.py"
        if not metadata_path.exists() or not solver_path.exists():
            continue

        try:
            metadata_json = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata = SolverMetadata.model_validate({**metadata_json, "source": str(plugin_dir)})

            module = _load_module_from_path(f"{plugin_dir.name}.solver", solver_path)
            factory = getattr(module, "get_solver", None)
            if not callable(factory):
                raise ValueError(f"{solver_path} must expose callable get_solver()")

            solver = factory()
            merged = _metadata_from_solver(solver, source=str(plugin_dir), quality=metadata.quality)
            discovered.append((solver, merged))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Skipping local plugin '{plugin_dir.name}': {exc}", stacklevel=2)

    return discovered


def _load_entrypoint_solvers() -> list[tuple[Solver, SolverMetadata]]:
    discovered: list[tuple[Solver, SolverMetadata]] = []
    eps = entry_points()
    selected = eps.select(group=ENTRYPOINT_GROUP) if hasattr(eps, "select") else eps.get(ENTRYPOINT_GROUP, [])

    for ep in selected:
        try:
            factory = ep.load()
            if not callable(factory):
                continue
            solver = factory()
            discovered.append((solver, _metadata_from_solver(solver, source=f"entrypoint:{ep.name}")))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Skipping entrypoint plugin '{ep.name}': {exc}", stacklevel=2)

    return discovered


def discover_solvers(
    *,
    plugin_root: str | Path = LOCAL_PLUGINS_DIR,
    use_entry_points: bool = True,
    reset_registry: bool = True,
) -> list[tuple[Solver, SolverMetadata]]:
    if reset_registry:
        clear_registry()

    root = Path(plugin_root)
    discovered = _load_local_solvers(root)
    if use_entry_points:
        discovered.extend(_load_entrypoint_solvers())

    for solver, metadata in discovered:
        register_solver(solver.name, solver, metadata)

    return discovered
