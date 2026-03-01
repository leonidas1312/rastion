"""Local registry management utilities."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import yaml

from rastion.registry.base import ProblemEntry, SolverEntry
from rastion.solvers.discovery import discover_plugins

_DEFAULT_HOME_DIR = ".rastion"
_REGISTRY_DIRNAME = "registry"
_PROBLEMS_DIRNAME = "problems"
_SOLVERS_DIRNAME = "solvers"
_RUNS_DIRNAME = "runs"
_DEFAULT_HUB_URL = "https://rastion-hub.onrender.com"

_OPEN_SOURCE_SOLVERS = {"baseline", "highs", "ortools", "scip", "neal", "qaoa"}

_BUILTIN_PROBLEM_METADATA: dict[str, dict[str, Any]] = {
    "knapsack": {
        "name": "knapsack",
        "version": "0.1.0",
        "author": "rastion",
        "tags": ["milp", "binary", "combinatorial"],
        "optimization_class": "MILP",
        "difficulty": "easy",
    },
    "maxcut": {
        "name": "maxcut",
        "version": "0.1.0",
        "author": "rastion",
        "tags": ["qubo", "graph", "sampling"],
        "optimization_class": "QUBO",
        "difficulty": "easy",
    },
    "portfolio": {
        "name": "portfolio",
        "version": "0.1.0",
        "author": "rastion",
        "tags": ["quadratic", "continuous", "finance"],
        "optimization_class": "QP",
        "difficulty": "medium",
    },
    "facility_location": {
        "name": "facility_location",
        "version": "0.1.0",
        "author": "rastion",
        "tags": ["milp", "logistics"],
        "optimization_class": "MILP",
        "difficulty": "medium",
    },
    "set_cover": {
        "name": "set_cover",
        "version": "0.1.0",
        "author": "rastion",
        "tags": ["milp", "binary", "covering"],
        "optimization_class": "MILP",
        "difficulty": "easy",
    },
    "tsp": {
        "name": "tsp",
        "version": "0.1.0",
        "author": "rastion",
        "tags": ["milp", "routing", "graph"],
        "optimization_class": "MILP",
        "difficulty": "hard",
    },
}


def rastion_home() -> Path:
    override = os.environ.get("RASTION_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / _DEFAULT_HOME_DIR).resolve()


def registry_root() -> Path:
    return rastion_home() / _REGISTRY_DIRNAME


def problems_root() -> Path:
    return registry_root() / _PROBLEMS_DIRNAME


def solvers_root() -> Path:
    return registry_root() / _SOLVERS_DIRNAME


def runs_root() -> Path:
    return rastion_home() / _RUNS_DIRNAME


def runs_file() -> Path:
    return runs_root() / "runs.jsonl"


def config_file() -> Path:
    return rastion_home() / "config.yaml"


def load_config() -> dict[str, Any]:
    init_registry(copy_examples=False)
    path = config_file()

    loaded: dict[str, Any] = {}
    if path.exists():
        parsed = read_yaml_file(path)
        if isinstance(parsed, dict):
            loaded = dict(parsed)

    changed, normalized = _apply_config_defaults(loaded)
    if changed:
        write_yaml_file(path, normalized)
    return normalized


def write_config(config: dict[str, Any]) -> None:
    _, normalized = _apply_config_defaults(dict(config))
    write_yaml_file(config_file(), normalized)


def hub_config() -> dict[str, Any]:
    cfg = load_config()
    hub = cfg.get("hub", {})
    if not isinstance(hub, dict):
        hub = {}
    return {
        "url": str(hub.get("url") or _DEFAULT_HUB_URL),
        "token": hub.get("token"),
    }


def set_hub_url(url: str) -> None:
    value = url.strip()
    if not value:
        raise ValueError("hub URL cannot be empty")

    cfg = load_config()
    hub = cfg.setdefault("hub", {})
    if not isinstance(hub, dict):
        hub = {}
        cfg["hub"] = hub
    hub["url"] = value
    write_config(cfg)


def set_hub_token(token: str | None) -> None:
    cfg = load_config()
    hub = cfg.setdefault("hub", {})
    if not isinstance(hub, dict):
        hub = {}
        cfg["hub"] = hub
    hub["token"] = token
    write_config(cfg)


def solver_hub_source(name: str) -> dict[str, Any] | None:
    init_registry(copy_examples=False)
    root = solvers_root()
    if not root.exists():
        return None

    preferred = root / name
    ordered: list[Path] = []
    if preferred.exists() and preferred.is_dir():
        ordered.append(preferred)
    ordered.extend(path for path in sorted(root.iterdir()) if path.is_dir() and path != preferred)

    for solver_dir in ordered:
        marker = solver_dir / "hub_source.yaml"
        if not marker.exists():
            continue
        loaded = read_yaml_file(marker)
        if not isinstance(loaded, dict):
            continue

        known_names = {solver_dir.name}
        for key in ("name", "plugin_name", "solver_name"):
            value = loaded.get(key)
            if isinstance(value, str) and value.strip():
                known_names.add(value.strip())

        if name in known_names:
            metadata = dict(loaded)
            metadata.setdefault("folder", solver_dir.name)
            return metadata

    return None


def _registry_marker() -> Path:
    return registry_root() / "init.py"


def _default_examples_root() -> Path:
    return Path(__file__).resolve().parents[2] / "examples"


def init_registry(
    root: str | Path | None = None,
    *,
    force: bool = False,
    copy_examples: bool = True,
) -> Path:
    base = Path(root).expanduser().resolve() if root is not None else rastion_home()
    reg = base / _REGISTRY_DIRNAME
    probs = reg / _PROBLEMS_DIRNAME
    solvers = reg / _SOLVERS_DIRNAME
    runs = base / _RUNS_DIRNAME

    reg.mkdir(parents=True, exist_ok=True)
    probs.mkdir(parents=True, exist_ok=True)
    solvers.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)

    _write_if_missing(base / "config.yaml", _default_config_yaml())
    _write_if_missing(reg / "init.py", _default_registry_init_py())
    _write_if_missing(runs / "runs.jsonl", "")

    if copy_examples:
        _seed_builtin_problems(probs, force=force)

    return base


def list_problems() -> list[ProblemEntry]:
    init_registry()

    entries: list[ProblemEntry] = []
    for root in sorted(problems_root().iterdir()):
        if not root.is_dir():
            continue

        metadata_path = root / "metadata.yaml"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            loaded = read_yaml_file(metadata_path)
            if isinstance(loaded, dict):
                metadata = dict(loaded)

        metadata.setdefault("name", root.name)
        metadata.setdefault("version", "0.1.0")
        metadata.setdefault("author", "unknown")
        metadata.setdefault("tags", [])
        metadata.setdefault("optimization_class", "unknown")
        metadata.setdefault("difficulty", "medium")
        metadata.setdefault("description", "")
        metadata["path"] = root

        entries.append(ProblemEntry.model_validate(metadata))

    return entries


def list_decision_plugins() -> list[ProblemEntry]:
    """List installed decision plugins."""
    return list_problems()


def list_solvers() -> list[SolverEntry]:
    plugins = discover_plugins()
    entries: list[SolverEntry] = []
    for name in sorted(plugins):
        plugin = plugins[name]
        cap = plugin.capabilities()
        entries.append(
            SolverEntry(
                name=plugin.name,
                version=plugin.version,
                author="rastion",
                tags=[mode.value for mode in sorted(cap.modes, key=lambda x: x.value)],
                description="Auto-discovered solver plugin",
                open_source=plugin.name in _OPEN_SOURCE_SOLVERS,
                capabilities={
                    "variable_types": sorted(v.value for v in cap.variable_types),
                    "objective_types": sorted(o.value for o in cap.objective_types),
                    "constraint_types": sorted(c.value for c in cap.constraint_types),
                    "modes": sorted(m.value for m in cap.modes),
                    "supports_miqp": cap.supports_miqp,
                },
            )
        )
    return entries


def add_problem(source: str | Path, name: str | None = None, *, overwrite: bool = False) -> Path:
    init_registry(copy_examples=False)

    src = Path(source).expanduser().resolve()
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"decision plugin source does not exist: {src}")

    target_name = name or _problem_name_from_source(src)
    dest = problems_root() / target_name

    if dest.exists():
        if not overwrite:
            raise FileExistsError(f"decision plugin '{target_name}' already exists in registry")
        shutil.rmtree(dest)

    dest.mkdir(parents=True, exist_ok=True)
    _copy_problem_folder(src, dest)
    _ensure_problem_files(dest)
    return dest


def add_decision_plugin(source: str | Path, name: str | None = None, *, overwrite: bool = False) -> Path:
    """Install a local decision plugin into the registry."""
    return add_problem(source, name=name, overwrite=overwrite)


def remove_problem(name: str) -> bool:
    init_registry(copy_examples=False)
    dest = problems_root() / name
    if not dest.exists():
        return False
    shutil.rmtree(dest)
    return True


def remove_decision_plugin(name: str) -> bool:
    """Remove a decision plugin from the local registry."""
    return remove_problem(name)


def export_problem(name: str, destination: str | Path) -> Path:
    init_registry(copy_examples=False)

    src = problems_root() / name
    if not src.exists():
        raise FileNotFoundError(f"decision plugin '{name}' not found in registry")

    dest = Path(destination).expanduser().resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"destination is not empty: {dest}")

    if not dest.exists():
        dest.mkdir(parents=True, exist_ok=True)

    _copy_problem_folder(src, dest)
    _ensure_problem_files(dest)
    return dest


def export_decision_plugin(name: str, destination: str | Path) -> Path:
    """Export a decision plugin folder from the local registry."""
    return export_problem(name, destination)


def install_problem(source: str | Path, *, overwrite: bool = False) -> Path:
    return add_problem(source, overwrite=overwrite)


def install_decision_plugin(source: str | Path, *, overwrite: bool = False) -> Path:
    """Install a local decision plugin folder into the registry."""
    return install_problem(source, overwrite=overwrite)


def install_solver_from_url(url: str, name: str | None = None, *, overwrite: bool = False) -> Path:
    """Download and install a solver plugin from a ZIP URL.

    The archive must contain a `solver.py` file somewhere in the extracted tree.
    """

    init_registry(copy_examples=False)
    source_url = url.strip()
    if not source_url:
        raise ValueError("solver URL cannot be empty")

    parsed = urllib.parse.urlparse(source_url)
    if parsed.scheme not in {"http", "https", "file"}:
        raise ValueError(f"unsupported URL scheme: {parsed.scheme!r}")

    with tempfile.TemporaryDirectory(prefix="rastion_solver_") as tmp:
        temp_root = Path(tmp)
        archive_path = temp_root / "solver.zip"
        extract_root = temp_root / "extract"
        extract_root.mkdir(parents=True, exist_ok=True)

        try:
            with urllib.request.urlopen(source_url, timeout=60) as response, archive_path.open("wb") as out:
                shutil.copyfileobj(response, out)
        except Exception as exc:
            raise RuntimeError(f"failed to download solver archive: {exc}") from exc

        try:
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(extract_root)
        except zipfile.BadZipFile as exc:
            raise ValueError("downloaded file is not a valid ZIP archive") from exc

        solver_file = _find_solver_file(extract_root)
        if solver_file is None:
            raise FileNotFoundError("solver archive does not contain solver.py")

        inferred_name = name or _infer_solver_name_from_url(source_url) or solver_file.parent.name
        target_name = _sanitize_solver_name(inferred_name)
        if not target_name:
            raise ValueError("could not derive a valid solver name")

        dest = solvers_root() / target_name
        if dest.exists():
            if not overwrite:
                raise FileExistsError(f"solver '{target_name}' already exists in registry")
            shutil.rmtree(dest)

        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(solver_file, dest / "solver.py")
        return dest


def read_yaml_file(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml_file(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _seed_builtin_problems(problems_dir: Path, *, force: bool) -> None:
    source_root = _default_examples_root()
    if not source_root.exists():
        return

    for source in sorted(source_root.iterdir()):
        if not source.is_dir():
            continue

        name = source.name
        if name not in _BUILTIN_PROBLEM_METADATA:
            continue

        dest = problems_dir / name
        if dest.exists() and not force:
            continue
        if dest.exists() and force:
            shutil.rmtree(dest)

        dest.mkdir(parents=True, exist_ok=True)
        _copy_example_problem(source, dest, name=name)


def _copy_example_problem(source_dir: Path, dest_dir: Path, *, name: str) -> None:
    spec_src = source_dir / "spec.json"
    if not spec_src.exists():
        raise FileNotFoundError(f"example is missing spec.json: {source_dir}")

    shutil.copy2(spec_src, dest_dir / "spec.json")

    instances_dir = dest_dir / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    default_instance = source_dir / "instance.json"
    if default_instance.exists():
        shutil.copy2(default_instance, instances_dir / "default.json")

    card_src = source_dir / "README.md"
    if card_src.exists():
        shutil.copy2(card_src, dest_dir / "problem_card.md")
    else:
        (dest_dir / "problem_card.md").write_text(_generated_problem_card(name), encoding="utf-8")

    metadata = dict(_BUILTIN_PROBLEM_METADATA.get(name, {}))
    metadata.setdefault("name", name)
    metadata.setdefault("version", "0.1.0")
    metadata.setdefault("author", "rastion")
    metadata.setdefault("tags", ["optimization"])
    metadata.setdefault("optimization_class", "unknown")
    metadata.setdefault("difficulty", "medium")
    write_yaml_file(dest_dir / "metadata.yaml", metadata)


def _copy_problem_folder(source_dir: Path, dest_dir: Path) -> None:
    spec_src = source_dir / "spec.json"
    if not spec_src.exists():
        raise FileNotFoundError(f"decision plugin folder missing spec.json: {source_dir}")
    shutil.copy2(spec_src, dest_dir / "spec.json")

    dest_instances = dest_dir / "instances"
    dest_instances.mkdir(parents=True, exist_ok=True)

    source_instances = source_dir / "instances"
    if source_instances.exists() and source_instances.is_dir():
        for item in sorted(source_instances.iterdir()):
            if item.is_file() and item.suffix.lower() in {".json", ".npz"}:
                shutil.copy2(item, dest_instances / item.name)
    else:
        legacy_instance = source_dir / "instance.json"
        if legacy_instance.exists():
            shutil.copy2(legacy_instance, dest_instances / "default.json")

    card_src = source_dir / "problem_card.md"
    if card_src.exists():
        shutil.copy2(card_src, dest_dir / "problem_card.md")
    else:
        readme_src = source_dir / "README.md"
        if readme_src.exists():
            shutil.copy2(readme_src, dest_dir / "problem_card.md")

    metadata_src = source_dir / "metadata.yaml"
    if metadata_src.exists():
        shutil.copy2(metadata_src, dest_dir / "metadata.yaml")

    decision_src = source_dir / "decision.yaml"
    if decision_src.exists():
        shutil.copy2(decision_src, dest_dir / "decision.yaml")


def _ensure_problem_files(problem_dir: Path) -> None:
    instances_dir = problem_dir / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    card_path = problem_dir / "problem_card.md"
    if not card_path.exists():
        card_path.write_text(_generated_problem_card(problem_dir.name), encoding="utf-8")

    metadata_path = problem_dir / "metadata.yaml"
    if not metadata_path.exists():
        metadata = dict(_BUILTIN_PROBLEM_METADATA.get(problem_dir.name, {}))
        metadata.setdefault("name", problem_dir.name)
        metadata.setdefault("version", "0.1.0")
        metadata.setdefault("author", "local")
        metadata.setdefault("tags", [])
        metadata.setdefault("optimization_class", "unknown")
        metadata.setdefault("difficulty", "medium")
        write_yaml_file(metadata_path, metadata)


def _problem_name_from_source(source: Path) -> str:
    metadata_path = source / "metadata.yaml"
    if metadata_path.exists():
        loaded = read_yaml_file(metadata_path)
        if isinstance(loaded, dict) and loaded.get("name"):
            return str(loaded["name"])
    return source.name


def _find_solver_file(extract_root: Path) -> Path | None:
    candidates = sorted(
        (path for path in extract_root.rglob("solver.py") if path.is_file()),
        key=lambda p: (len(p.parts), str(p)),
    )
    return candidates[0] if candidates else None


def _infer_solver_name_from_url(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    segments = [segment for segment in parsed.path.split("/") if segment]
    if not segments:
        return None

    for marker in ("archive", "zip"):
        if marker in segments:
            idx = segments.index(marker)
            if idx >= 1:
                return segments[idx - 1]

    last = segments[-1]
    if last.endswith(".zip"):
        return last[: -len(".zip")]
    return last


def _sanitize_solver_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return normalized.strip("._-")


def _write_if_missing(path: Path, content: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _generated_problem_card(name: str) -> str:
    return (
        f"# {name}\n\n"
        "This decision plugin was installed into the local Rastion registry.\n\n"
        "## Variables\n"
        "See `spec.json` for variable definitions.\n\n"
        "## Constraints\n"
        "See `spec.json` + instance payloads in `instances/`.\n\n"
        "## Recommended Solvers\n"
        "Start with `highs` or `ortools` depending compatibility.\n"
    )


def _default_config_yaml() -> str:
    return yaml.safe_dump(_default_config_data(), sort_keys=False)


def _default_config_data() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "registry": {
            "provider": "local",
            "remote_hub": None,
            "auto_sync": False,
        },
        "hub": {
            "url": _DEFAULT_HUB_URL,
            "token": None,
        },
    }


def _apply_config_defaults(config: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    defaults = _default_config_data()
    normalized = dict(config)
    changed = False

    if "schema_version" not in normalized:
        normalized["schema_version"] = defaults["schema_version"]
        changed = True

    registry = normalized.get("registry")
    if not isinstance(registry, dict):
        registry = {}
        normalized["registry"] = registry
        changed = True
    default_registry = defaults["registry"]
    for key, value in default_registry.items():
        if key not in registry:
            registry[key] = value
            changed = True

    hub = normalized.get("hub")
    if not isinstance(hub, dict):
        hub = {}
        normalized["hub"] = hub
        changed = True
    default_hub = defaults["hub"]
    if not isinstance(hub.get("url"), str) or not str(hub.get("url")).strip():
        hub["url"] = default_hub["url"]
        changed = True
    if "token" not in hub:
        hub["token"] = default_hub["token"]
        changed = True

    return changed, normalized


def _default_registry_init_py() -> str:
    return '"""Local Rastion registry marker for future remote sync support."""\n'
