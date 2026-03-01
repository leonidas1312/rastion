"""Structured agent-run orchestration for local optimization workflows."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from rastion.backends.local import LocalBackend
from rastion.compile.normalize import compile_to_ir
from rastion.core.data import InstanceData
from rastion.core.run_record import append_run_record, create_run_record
from rastion.core.solution import Solution, SolutionStatus
from rastion.core.spec import ProblemSpec
from rastion.core.validate import validate_problem_and_instance
from rastion.registry.loader import DecisionPlugin
from rastion.registry.manager import init_registry, install_decision_plugin, install_solver_from_url, runs_root
from rastion.solvers.discovery import discover_plugins
from rastion.solvers.matching import auto_select_solver, compatible_plugins


def run_agent_request_file(
    request_path: str | Path,
    *,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Run a structured agent request end-to-end and return result payload."""
    source = Path(request_path).expanduser().resolve()
    request = _read_request(source)
    _resolve_relative_request_paths(request, source.parent)
    return run_agent_request(request, output_json=output_json)


def run_agent_request(
    request: dict[str, Any],
    *,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Run a structured agent request end-to-end and return result payload."""
    init_registry()

    install_overwrite = bool(request.get("install_overwrite", True))
    decision_plugin_name, instance_name, spec, instance = _prepare_decision_plugin(
        request,
        overwrite=install_overwrite,
    )

    validation = validate_problem_and_instance(spec, instance)
    if not validation.valid:
        raise ValueError("validation failed: " + "; ".join(validation.errors))

    ir_model = compile_to_ir(spec, instance)
    solver_section = request.get("solver", {})
    if solver_section is None:
        solver_section = {}
    if not isinstance(solver_section, dict):
        raise ValueError("'solver' must be an object when provided")

    solver_name = str(solver_section.get("name") or "auto")
    preferred = _coerce_str_list(solver_section.get("preferred"))
    solver_config = _coerce_solver_config(solver_section.get("config"))
    solver_config.setdefault("time_limit", 30.0)

    policy_section = request.get("solver_policy", {})
    if policy_section is None:
        policy_section = {}
    if not isinstance(policy_section, dict):
        raise ValueError("'solver_policy' must be an object when provided")

    allow_remote_install = bool(policy_section.get("allow_remote_install", False))
    solver_urls = _coerce_str_list(policy_section.get("install_urls"))
    solver_overwrite = bool(policy_section.get("overwrite", False))

    plugins = discover_plugins()
    installed_from_url: list[str] = []

    if solver_name == "auto":
        candidates = compatible_plugins(ir_model, plugins, preferred=preferred)
        if not candidates and allow_remote_install and solver_urls:
            installed_from_url = _install_solver_urls(solver_urls, overwrite=solver_overwrite)
            plugins = discover_plugins()
            candidates = compatible_plugins(ir_model, plugins, preferred=preferred)

        if not candidates:
            try:
                auto_select_solver(ir_model, plugins, preferred=preferred)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            raise ValueError("no compatible solver available")

        solver_plugin, solution = _run_auto_candidates(candidates, ir_model, solver_config)
    else:
        if solver_name not in plugins and allow_remote_install and solver_urls:
            installed_from_url = _install_solver_urls(solver_urls, overwrite=solver_overwrite)
            plugins = discover_plugins()

        if solver_name not in plugins:
            raise ValueError(
                f"requested solver '{solver_name}' is not installed (available: {', '.join(sorted(plugins)) or 'none'})"
            )

        solver_plugin = plugins[solver_name]
        solution = _run_plugin(solver_plugin, ir_model, solver_config)

    runs_dir_raw = request.get("runs_dir")
    if isinstance(runs_dir_raw, str) and runs_dir_raw.strip():
        runs_dir = Path(runs_dir_raw).expanduser()
    else:
        runs_dir = runs_root()

    run_record = create_run_record(
        spec=spec,
        instance=instance,
        solution=solution,
        solver_name=solver_plugin.name,
        solver_version=solver_plugin.version,
        solver_config=solver_config,
    )
    run_file = append_run_record(run_record, runs_dir=runs_dir)

    result = {
        "decision_plugin": decision_plugin_name,
        "instance": instance_name,
        "solver": {
            "name": solver_plugin.name,
            "version": solver_plugin.version,
        },
        "status": solution.status.value,
        "objective_value": solution.objective_value,
        "runtime_s": solution.metadata.get("runtime_s"),
        "error_message": solution.error_message,
        "installed_solver_folders": installed_from_url,
        "run_record_file": str(run_file),
    }

    output_path = output_json or request.get("output_json")
    if isinstance(output_path, str) and output_path.strip():
        resolved = Path(output_path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result["output_json"] = str(resolved)
    elif isinstance(output_path, Path):
        resolved = output_path.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result["output_json"] = str(resolved)

    return result


def _prepare_decision_plugin(
    request: dict[str, Any],
    *,
    overwrite: bool,
) -> tuple[str, str, ProblemSpec, InstanceData]:
    decision_plugin_path = request.get("decision_plugin_path")
    inline_decision_plugin = request.get("decision_plugin")
    instance_name = str(request.get("instance_name") or "default")

    source_path = decision_plugin_path
    inline_payload = inline_decision_plugin

    if source_path is not None and inline_payload is not None:
        raise ValueError("provide either '*_path' or inline payload, not both")
    if source_path is None and inline_payload is None:
        raise ValueError("request must provide decision_plugin_path or decision_plugin")

    if isinstance(source_path, str):
        installed = install_decision_plugin(source_path, overwrite=overwrite)
    elif source_path is not None:
        raise ValueError("'decision_plugin_path' must be a string path")
    else:
        if not isinstance(inline_payload, dict):
            raise ValueError("'decision_plugin' must be an object")
        installed = _install_inline_decision_plugin(inline_payload, overwrite=overwrite)

    requested_name_raw = request.get("decision_plugin_name")
    if requested_name_raw is not None:
        requested_name = str(requested_name_raw).strip()
        if requested_name and requested_name != installed.name:
            raise ValueError(
                f"decision_plugin_name '{requested_name}' does not match installed plugin '{installed.name}'"
            )
    plugin_name = installed.name
    plugin = DecisionPlugin.from_registry(plugin_name)
    spec = plugin.spec
    instance = plugin.load_instance(instance_name)
    return plugin_name, instance_name, spec, instance


def _install_inline_decision_plugin(decision_plugin: dict[str, Any], *, overwrite: bool) -> Path:
    spec = decision_plugin.get("spec")
    instance = decision_plugin.get("instance")
    if not isinstance(spec, dict):
        raise ValueError("inline decision_plugin is missing object field 'spec'")
    if not isinstance(instance, dict):
        raise ValueError("inline decision_plugin is missing object field 'instance'")

    metadata = decision_plugin.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("inline decision_plugin field 'metadata' must be an object")
    card = decision_plugin.get("problem_card")
    if card is not None and not isinstance(card, str):
        raise ValueError("inline decision_plugin field 'problem_card' must be a string")

    with tempfile.TemporaryDirectory(prefix="rastion_agent_inline_") as tmp:
        root = Path(tmp)
        (root / "instances").mkdir(parents=True, exist_ok=True)
        (root / "spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (root / "instances" / "default.json").write_text(
            json.dumps(instance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if metadata is not None:
            (root / "metadata.yaml").write_text(_to_simple_yaml(metadata), encoding="utf-8")
        if card is not None:
            (root / "problem_card.md").write_text(card, encoding="utf-8")
        return install_decision_plugin(root, overwrite=overwrite)


def _to_simple_yaml(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in payload.items():
        if isinstance(value, str):
            lines.append(f'{key}: "{value}"')
        elif isinstance(value, bool):
            lines.append(f"{key}: {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key}: {value}")
        elif isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {json.dumps(value)}")
    return "\n".join(lines) + "\n"


def _install_solver_urls(urls: list[str], *, overwrite: bool) -> list[str]:
    installed: list[str] = []
    for url in urls:
        target = install_solver_from_url(url, overwrite=overwrite)
        installed.append(target.name)
    return installed


def _run_auto_candidates(
    candidates: list[object],
    ir_model: object,
    solver_config: dict[str, object],
) -> tuple[object, Solution]:
    last_error: Exception | None = None
    selected = None
    for plugin in candidates:
        selected = plugin
        try:
            solution = _run_plugin(plugin, ir_model, solver_config)
        except Exception as exc:
            last_error = exc
            continue
        if solution.status != SolutionStatus.ERROR:
            return plugin, solution
        last_error = RuntimeError(solution.error_message or "solver returned ERROR status")

    if selected is None:
        raise ValueError("no compatible solver candidates")
    if last_error is not None:
        raise RuntimeError(f"all compatible solvers failed: {last_error}") from last_error
    raise RuntimeError("all compatible solvers failed")


def _run_plugin(plugin: object, ir_model: object, solver_config: dict[str, object]) -> Solution:
    backend = LocalBackend()
    try:
        return backend.run(plugin, ir_model, solver_config)
    except Exception as exc:
        return Solution(
            status=SolutionStatus.ERROR,
            objective_value=None,
            primal_values={},
            metadata={
                "runtime_s": 0.0,
                "solver_name": str(getattr(plugin, "name", "unknown")),
                "solver_version": str(getattr(plugin, "version", "unknown")),
            },
            error_message=str(exc),
        )


def _read_request(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"agent request file not found: {source}")
    loaded = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("agent request JSON must be an object")
    return loaded


def _resolve_relative_request_paths(request: dict[str, Any], base_dir: Path) -> None:
    for key in ("decision_plugin_path", "runs_dir", "output_json"):
        raw = request.get(key)
        if not isinstance(raw, str):
            continue
        stripped = raw.strip()
        if not stripped:
            continue
        path = Path(stripped).expanduser()
        if path.is_absolute():
            request[key] = str(path)
            continue
        primary = (base_dir / path).resolve()
        if key == "decision_plugin_path" and not primary.exists():
            alternate = (base_dir.parent / path).resolve()
            if alternate.exists():
                request[key] = str(alternate)
                continue
        request[key] = str(primary)


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    out.append(stripped)
        return out
    raise ValueError("expected a list of strings")


def _coerce_solver_config(value: Any) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items() if v is not None}
    raise ValueError("'solver.config' must be an object")
