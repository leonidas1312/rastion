"""Plugin discovery for built-in and optional solver adapters."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path

from rastion.solvers.base import SolverPlugin

PluginFactory = Callable[[], SolverPlugin | None]

_PLUGIN_FACTORIES: list[PluginFactory] = []
_BUILTINS_REGISTERED = False


def register_plugin(factory: PluginFactory) -> None:
    _PLUGIN_FACTORIES.append(factory)


def _register_builtin_plugins() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    from rastion.solvers.baseline import get_plugin as baseline_plugin
    from rastion.solvers.highs import get_plugin as highs_plugin
    from rastion.solvers.neal import get_plugin as neal_plugin
    from rastion.solvers.ortools_mip import get_plugin as ortools_plugin
    from rastion.solvers.qaoa import get_plugin as qaoa_plugin
    from rastion.solvers.scip import get_plugin as scip_plugin

    register_plugin(baseline_plugin)
    register_plugin(highs_plugin)
    register_plugin(ortools_plugin)
    register_plugin(scip_plugin)
    register_plugin(neal_plugin)
    register_plugin(qaoa_plugin)
    _BUILTINS_REGISTERED = True


def discover_plugins() -> dict[str, SolverPlugin]:
    _register_builtin_plugins()
    plugins: dict[str, SolverPlugin] = {}

    for factory in _PLUGIN_FACTORIES:
        try:
            plugin = factory()
        except ImportError:
            plugin = None
        if plugin is None:
            continue
        plugins[plugin.name] = plugin

    plugins.update(_discover_external_solvers())

    return plugins


def _discover_external_solvers() -> dict[str, SolverPlugin]:
    """Load external solvers from the local registry."""
    from rastion.registry.manager import solvers_root

    discovered: dict[str, SolverPlugin] = {}
    root = solvers_root()
    if not root.exists():
        return discovered

    for solver_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if not solver_dir.is_dir():
            continue
        solver_file = solver_dir / "solver.py"
        if not solver_file.exists() or not solver_file.is_file():
            continue

        plugin = _load_external_plugin(solver_dir.name, solver_file)
        if plugin is None:
            continue
        discovered[plugin.name] = plugin

    return discovered


def _load_external_plugin(namespace: str, solver_file: Path) -> SolverPlugin | None:
    module_name = f"rastion_external_{namespace}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, solver_file)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        get_plugin = getattr(module, "get_plugin", None)
        if get_plugin is None:
            return None
        plugin = get_plugin()
        if plugin is None:
            return None
        if not isinstance(plugin, SolverPlugin):
            return None
        return plugin
    except Exception:
        return None
