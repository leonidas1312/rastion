"""Registry public API."""

from rastion.registry.loader import AutoSolver, DecisionPlugin, Problem, Solver
from rastion.registry.manager import (
    add_decision_plugin,
    export_decision_plugin,
    init_registry,
    install_decision_plugin,
    install_solver_from_url,
    list_decision_plugins,
    list_solvers,
    remove_decision_plugin,
)

__all__ = [
    "AutoSolver",
    "DecisionPlugin",
    "Problem",
    "Solver",
    "add_decision_plugin",
    "export_decision_plugin",
    "init_registry",
    "install_decision_plugin",
    "install_solver_from_url",
    "list_decision_plugins",
    "list_solvers",
    "remove_decision_plugin",
]
