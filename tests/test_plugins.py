from pathlib import Path

from rastion.plugins.discovery import discover_solvers
from rastion.plugins.registry import clear_registry, list_registered_solvers


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_local_plugin_discovery_registers_demo_plugins() -> None:
    clear_registry()
    discovered = discover_solvers(
        plugin_root=REPO_ROOT / "plugins_local",
        use_entry_points=False,
        reset_registry=True,
    )

    names = {solver.name for solver, _ in discovered}
    assert {"tabu", "simulated_annealing"}.issubset(names)

    registry_names = {item.solver.name for item in list_registered_solvers()}
    assert {"tabu", "simulated_annealing"}.issubset(registry_names)
