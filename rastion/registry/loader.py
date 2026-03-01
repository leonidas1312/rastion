"""Decision plugin and solver loaders for the local registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rastion.compile.normalize import compile_to_ir
from rastion.core.data import InstanceData
from rastion.core.ir import IRModel
from rastion.core.solution import Solution
from rastion.core.spec import ProblemSpec, VariableType
from rastion.registry.manager import init_registry, problems_root, read_yaml_file
from rastion.solvers.base import (
    CapabilitySet,
    ConstraintType,
    ObjectiveType,
    SolveMode,
    SolverPlugin,
)
from rastion.solvers.discovery import discover_plugins
from rastion.solvers.matching import auto_select_solver

_OPEN_SOURCE_SOLVERS = {"baseline", "highs", "ortools", "scip", "neal", "qaoa"}
_KNOWN_SOLVERS: dict[str, CapabilitySet] = {
    "baseline": CapabilitySet(
        variable_types={VariableType.BINARY},
        objective_types={ObjectiveType.LINEAR},
        constraint_types={ConstraintType.LINEAR},
        modes={SolveMode.SOLVE},
    ),
    "highs": CapabilitySet(
        variable_types={VariableType.BINARY, VariableType.INTEGER, VariableType.CONTINUOUS},
        objective_types={ObjectiveType.LINEAR, ObjectiveType.QUADRATIC},
        constraint_types={ConstraintType.LINEAR},
        modes={SolveMode.SOLVE},
        supports_miqp=False,
    ),
    "ortools": CapabilitySet(
        variable_types={VariableType.BINARY, VariableType.INTEGER, VariableType.CONTINUOUS},
        objective_types={ObjectiveType.LINEAR},
        constraint_types={ConstraintType.LINEAR},
        modes={SolveMode.SOLVE},
    ),
    "scip": CapabilitySet(
        variable_types={VariableType.BINARY, VariableType.INTEGER, VariableType.CONTINUOUS},
        objective_types={ObjectiveType.LINEAR},
        constraint_types={ConstraintType.LINEAR},
        modes={SolveMode.SOLVE},
    ),
    "neal": CapabilitySet(
        variable_types={VariableType.BINARY},
        objective_types={ObjectiveType.QUBO},
        constraint_types=set(),
        modes={SolveMode.SAMPLE},
    ),
    "qaoa": CapabilitySet(
        variable_types={VariableType.BINARY},
        objective_types={ObjectiveType.QUBO},
        constraint_types=set(),
        modes={SolveMode.SAMPLE},
    ),
}


class _UnavailableSolverPlugin(SolverPlugin):
    def __init__(self, name: str, capabilities: CapabilitySet) -> None:
        self.name = name
        self.version = "unavailable"
        self._capabilities = capabilities

    def capabilities(self) -> CapabilitySet:
        return self._capabilities

    def solve(self, ir_model: IRModel, config: dict[str, object], backend: object) -> Solution:
        raise RuntimeError(
            f"solver '{self.name}' is known but not installed. Install extras before calling solve()."
        )


class DecisionPlugin:
    """Load a decision plugin from registry or local path."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser().resolve()
        if not self._root.exists() or not self._root.is_dir():
            raise FileNotFoundError(f"decision plugin folder not found: {self._root}")

        if not (self._root / "spec.json").exists():
            raise FileNotFoundError(f"decision plugin folder is missing spec.json: {self._root}")

        self._spec: ProblemSpec | None = None
        self._metadata: dict[str, Any] | None = None
        self._card: str | None = None

    @staticmethod
    def from_registry(name: str) -> "DecisionPlugin":
        """Load a decision plugin from local registry by name."""
        init_registry()
        root = problems_root() / name
        if not root.exists():
            raise FileNotFoundError(f"decision plugin '{name}' not found in local registry")
        return DecisionPlugin(root)

    @staticmethod
    def from_local(path: str | Path) -> "DecisionPlugin":
        """Load a decision plugin from a local folder."""
        root = Path(path).expanduser().resolve()
        if root.is_file():
            root = root.parent
        return DecisionPlugin(root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def spec(self) -> ProblemSpec:
        if self._spec is None:
            self._spec = ProblemSpec.from_json_file(self._root / "spec.json")
        return self._spec

    @property
    def instances(self) -> list[str]:
        instances_dir = self._root / "instances"
        if not instances_dir.exists():
            legacy = self._root / "instance.json"
            return ["default"] if legacy.exists() else []

        names: set[str] = set()
        for path in instances_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {".json", ".npz"}:
                names.add(path.stem)
        return sorted(names)

    @property
    def card(self) -> str:
        if self._card is not None:
            return self._card

        card_path = self._root / "problem_card.md"
        if card_path.exists():
            self._card = card_path.read_text(encoding="utf-8")
        else:
            self._card = f"# {self.spec.name}\n\nNo decision plugin card available."
        return self._card

    @property
    def metadata(self) -> dict[str, Any]:
        if self._metadata is not None:
            return dict(self._metadata)

        metadata_path = self._root / "metadata.yaml"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            loaded = read_yaml_file(metadata_path)
            if isinstance(loaded, dict):
                metadata = dict(loaded)

        metadata.setdefault("name", self._root.name)
        metadata.setdefault("version", "0.1.0")
        metadata.setdefault("author", "unknown")
        metadata.setdefault("tags", [])
        metadata.setdefault("optimization_class", "unknown")
        metadata.setdefault("difficulty", "medium")
        self._metadata = metadata
        return dict(self._metadata)

    def load_instance(self, name: str = "default") -> InstanceData:
        """Load specific instance by name."""
        instances_dir = self._root / "instances"
        if instances_dir.exists():
            json_path = instances_dir / f"{name}.json"
            if json_path.exists():
                return InstanceData.from_json_file(json_path)

            npz_path = instances_dir / f"{name}.npz"
            if npz_path.exists():
                return InstanceData.from_npz_file(npz_path)

            raise FileNotFoundError(
                f"instance '{name}' not found under {instances_dir}; available: {', '.join(self.instances) or 'none'}"
            )

        # Legacy local folder support
        legacy_json = self._root / "instance.json"
        if legacy_json.exists() and name == "default":
            return InstanceData.from_json_file(legacy_json)

        raise FileNotFoundError(f"instance '{name}' not found in {self._root}")

    def all_instances(self) -> dict[str, InstanceData]:
        """Load all instances."""
        names = self.instances
        if not names and (self._root / "instance.json").exists():
            return {"default": InstanceData.from_json_file(self._root / "instance.json")}
        return {name: self.load_instance(name) for name in names}


class Solver:
    """Load a solver plugin."""

    def __init__(self, plugin: SolverPlugin) -> None:
        self._plugin = plugin

    @staticmethod
    def from_name(name: str) -> "Solver":
        """Load solver by name (e.g., 'highs', 'baseline')."""
        plugins = discover_plugins()
        if name in plugins:
            return Solver(plugins[name])
        if name in _KNOWN_SOLVERS:
            return Solver(_UnavailableSolverPlugin(name, _KNOWN_SOLVERS[name]))

        available = sorted(set(plugins).union(_KNOWN_SOLVERS))
        raise ValueError(f"solver '{name}' is not available (available: {', '.join(available) or 'none'})")

    @staticmethod
    def available() -> list[str]:
        """List all available solver names."""
        return sorted(discover_plugins())

    @property
    def plugin(self) -> SolverPlugin:
        return self._plugin

    @property
    def capabilities(self) -> CapabilitySet:
        return self._plugin.capabilities()

    @property
    def card(self) -> str:
        cap = self.capabilities
        vars_supported = ", ".join(sorted(v.value for v in cap.variable_types)) or "none"
        obj_supported = ", ".join(sorted(o.value for o in cap.objective_types)) or "none"
        cons_supported = ", ".join(sorted(c.value for c in cap.constraint_types)) or "none"
        modes = ", ".join(sorted(m.value for m in cap.modes)) or "none"
        source = "open-source" if self._plugin.name in _OPEN_SOURCE_SOLVERS else "closed-source/unknown"

        return (
            f"# Solver: {self._plugin.name}\n\n"
            f"- Version: {self._plugin.version}\n"
            f"- License: {source}\n"
            f"- Variable types: {vars_supported}\n"
            f"- Objective types: {obj_supported}\n"
            f"- Constraint types: {cons_supported}\n"
            f"- Modes: {modes}\n"
            f"- Supports MIQP: {cap.supports_miqp}\n"
        )


class AutoSolver:
    """Auto-select solver based on decision plugin shape."""

    @staticmethod
    def from_decision_plugin(decision_plugin: DecisionPlugin) -> Solver:
        """Auto-select best solver for a decision plugin."""
        if "default" in decision_plugin.instances:
            instance_name = "default"
        elif decision_plugin.instances:
            instance_name = decision_plugin.instances[0]
        else:
            instance_name = None
        if instance_name is None:
            raise ValueError(
                f"decision plugin '{decision_plugin.metadata.get('name', decision_plugin.root.name)}' has no instances"
            )

        instance = decision_plugin.load_instance(instance_name)
        ir_model = compile_to_ir(decision_plugin.spec, instance)
        plugins = discover_plugins()
        if not plugins:
            raise ValueError("no solver plugins available")
        selected = auto_select_solver(ir_model, plugins)
        return Solver(selected)

    @staticmethod
    def from_problem(problem: "DecisionPlugin") -> Solver:
        """Backward-compatible alias for from_decision_plugin()."""
        return AutoSolver.from_decision_plugin(problem)

    @staticmethod
    def from_preferences(
        preferred: list[str] | None = None,
        budget: str | None = None,
        open_source: bool = True,
    ) -> Solver:
        """Auto-select solver with preferences."""
        plugins = discover_plugins()
        if not plugins:
            raise ValueError("no solver plugins available")

        available = dict(plugins)
        if open_source:
            available = {name: plugin for name, plugin in available.items() if name in _OPEN_SOURCE_SOLVERS}
            if not available:
                raise ValueError("no open-source solver plugins available")

        preferred = preferred or []
        for name in preferred:
            if name in available:
                return Solver(available[name])

        budget_s = _parse_budget_seconds(budget)
        if budget_s is not None and budget_s <= 10:
            for candidate in ("highs", "ortools", "baseline", "neal", "qaoa", "scip"):
                if candidate in available:
                    return Solver(available[candidate])

        for candidate in ("highs", "ortools", "scip", "baseline", "neal", "qaoa"):
            if candidate in available:
                return Solver(available[candidate])

        first = sorted(available)[0]
        return Solver(available[first])


Problem = DecisionPlugin


def _parse_budget_seconds(budget: str | None) -> float | None:
    if budget is None:
        return None

    text = budget.strip().lower()
    if not text:
        return None

    try:
        if text.endswith("ms"):
            return float(text[:-2]) / 1000.0
        if text.endswith("s"):
            return float(text[:-1])
        if text.endswith("m"):
            return float(text[:-1]) * 60.0
        return float(text)
    except ValueError:
        return None
