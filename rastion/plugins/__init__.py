"""Plugin discovery and registry compatibility exports."""

__all__ = [
    "discover_solvers",
    "register_solver",
    "get_solver",
    "list_registered_solvers",
    "clear_registry",
]


def __getattr__(name: str):
    if name == "discover_solvers":
        from .discovery import discover_solvers

        return discover_solvers
    if name in {"register_solver", "get_solver", "list_registered_solvers", "clear_registry"}:
        from .registry import clear_registry, get_solver, list_registered_solvers, register_solver

        mapping = {
            "register_solver": register_solver,
            "get_solver": get_solver,
            "list_registered_solvers": list_registered_solvers,
            "clear_registry": clear_registry,
        }
        return mapping[name]
    raise AttributeError(name)
