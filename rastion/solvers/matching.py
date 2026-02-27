"""Capability matching, scoring, and auto solver selection."""

from __future__ import annotations

from dataclasses import dataclass

from rastion.core.ir import IRModel
from rastion.core.spec import IRTarget, VariableType
from rastion.solvers.base import (
    CapabilitySet,
    ConstraintType,
    ObjectiveType,
    ProblemRequirements,
    SolveMode,
    SolverPlugin,
)

_OPEN_SOURCE_SOLVERS = {"baseline", "highs", "ortools", "scip", "neal", "qaoa"}
_DETERMINISTIC_SOLVERS = {"baseline", "highs", "ortools", "scip"}
_SOLVER_BASE_PREFERENCE = {
    "highs": 40,
    "ortools": 35,
    "scip": 30,
    "baseline": 10,
    "neal": 25,
    "qaoa": 20,
}


@dataclass(frozen=True)
class MatchReport:
    plugin_name: str
    matched: bool
    score: int
    missing: list[str]


def requirements_from_ir(ir_model: IRModel) -> ProblemRequirements:
    objective_type = ObjectiveType.LINEAR
    if ir_model.qubo is not None or ir_model.target == IRTarget.QUBO:
        objective_type = ObjectiveType.QUBO
    elif ir_model.objective.q_v:
        objective_type = ObjectiveType.QUADRATIC

    has_constraints = ir_model.constraints is not None and ir_model.constraints.num_rows > 0
    constraint_types = {ConstraintType.LINEAR} if has_constraints else set()
    variable_types = {v.vartype for v in ir_model.variables}

    mode = SolveMode.SOLVE
    if objective_type == ObjectiveType.QUBO:
        mode = SolveMode.SAMPLE

    has_integer = bool(variable_types.intersection({VariableType.BINARY, VariableType.INTEGER}))

    return ProblemRequirements(
        variable_types=variable_types,
        objective_type=objective_type,
        constraint_types=constraint_types,
        mode=mode,
        has_integer_variables=has_integer,
    )


def required_capabilities_summary(requirements: ProblemRequirements) -> str:
    variable_types = ",".join(sorted(v.value for v in requirements.variable_types)) or "none"
    constraint_types = ",".join(sorted(c.value for c in requirements.constraint_types)) or "none"
    return (
        f"objective={requirements.objective_type.value}, "
        f"variables={variable_types}, "
        f"constraints={constraint_types}, "
        f"mode={requirements.mode.value}, "
        f"requires_miqp={requirements.objective_type == ObjectiveType.QUADRATIC and requirements.has_integer_variables}"
    )


def missing_capabilities(requirements: ProblemRequirements, capabilities: CapabilitySet) -> list[str]:
    missing: list[str] = []

    missing_variable_types = sorted(v.value for v in (requirements.variable_types - capabilities.variable_types))
    if missing_variable_types:
        missing.append("variable_types: missing " + ",".join(missing_variable_types))

    if requirements.objective_type not in capabilities.objective_types:
        missing.append(f"objective_type: requires {requirements.objective_type.value}")

    missing_constraint_types = sorted(
        c.value for c in (requirements.constraint_types - capabilities.constraint_types)
    )
    if missing_constraint_types:
        missing.append("constraint_types: missing " + ",".join(missing_constraint_types))

    if requirements.mode not in capabilities.modes:
        missing.append(f"mode: requires {requirements.mode.value}")

    if (
        requirements.objective_type == ObjectiveType.QUADRATIC
        and requirements.has_integer_variables
        and not capabilities.supports_miqp
    ):
        missing.append("supports_miqp: required for integer quadratic models")

    return missing


def match(requirements: ProblemRequirements, capabilities: CapabilitySet) -> bool:
    return len(missing_capabilities(requirements, capabilities)) == 0


def plugin_score(
    requirements: ProblemRequirements,
    plugin_name: str,
    capabilities: CapabilitySet,
    preferred_rank: int | None = None,
) -> int:
    missing = missing_capabilities(requirements, capabilities)

    score = 0
    if not missing:
        score += 1000
    else:
        score += max(0, 240 - 40 * len(missing))

    score += 15 * len(requirements.variable_types.intersection(capabilities.variable_types))
    score += 25 if requirements.objective_type in capabilities.objective_types else 0
    score += 10 * len(requirements.constraint_types.intersection(capabilities.constraint_types))
    score += 10 if requirements.mode in capabilities.modes else 0

    if plugin_name in _OPEN_SOURCE_SOLVERS:
        score += 10
    if requirements.objective_type != ObjectiveType.QUBO and plugin_name in _DETERMINISTIC_SOLVERS:
        score += 10

    score += _SOLVER_BASE_PREFERENCE.get(plugin_name, 0)

    if preferred_rank is not None:
        score += 500 - (10 * preferred_rank)

    return score


def match_report(
    requirements: ProblemRequirements,
    plugin_name: str,
    capabilities: CapabilitySet,
    preferred_rank: int | None = None,
) -> MatchReport:
    missing = missing_capabilities(requirements, capabilities)
    return MatchReport(
        plugin_name=plugin_name,
        matched=not missing,
        score=plugin_score(
            requirements,
            plugin_name,
            capabilities,
            preferred_rank=preferred_rank,
        ),
        missing=missing,
    )


def ranked_plugins(
    ir_model: IRModel,
    plugins: dict[str, SolverPlugin],
    preferred: list[str] | None = None,
) -> list[tuple[SolverPlugin, MatchReport]]:
    requirements = requirements_from_ir(ir_model)
    preferred = preferred or []
    preferred_rank = {name: idx for idx, name in enumerate(preferred)}

    ranked: list[tuple[SolverPlugin, MatchReport]] = []
    for name in sorted(plugins):
        plugin = plugins[name]
        report = match_report(
            requirements,
            plugin_name=name,
            capabilities=plugin.capabilities(),
            preferred_rank=preferred_rank.get(name),
        )
        ranked.append((plugin, report))

    ranked.sort(key=lambda item: item[1].score, reverse=True)
    return ranked


def compatible_plugins(
    ir_model: IRModel,
    plugins: dict[str, SolverPlugin],
    preferred: list[str] | None = None,
) -> list[SolverPlugin]:
    preferred = preferred or []
    requirements = requirements_from_ir(ir_model)

    ordered: list[SolverPlugin] = []
    selected_names: set[str] = set()

    for name in preferred:
        plugin = plugins.get(name)
        if plugin is None:
            continue
        if match(requirements, plugin.capabilities()):
            ordered.append(plugin)
            selected_names.add(name)

    ranked = ranked_plugins(ir_model, plugins, preferred=preferred)
    for plugin, report in ranked:
        if report.matched and plugin.name not in selected_names:
            ordered.append(plugin)

    return ordered


def auto_select_solver(
    ir_model: IRModel,
    plugins: dict[str, SolverPlugin],
    preferred: list[str] | None = None,
) -> SolverPlugin:
    preferred = preferred or []
    requirements = requirements_from_ir(ir_model)

    for name in preferred:
        plugin = plugins.get(name)
        if plugin is None:
            continue
        if match(requirements, plugin.capabilities()):
            return plugin

    ranked = ranked_plugins(ir_model, plugins, preferred=preferred)
    for plugin, report in ranked:
        if report.matched:
            return plugin

    required = required_capabilities_summary(requirements)
    closest = ranked[:3]
    if not closest:
        raise ValueError(f"No solver plugins available. Required capabilities: {required}")

    lines = [f"No compatible solver found. Required capabilities: {required}", "Closest candidates:"]
    for plugin, report in closest:
        missing = "; ".join(report.missing) if report.missing else "none"
        lines.append(f"- {plugin.name}: score={report.score}; missing: {missing}")

    raise ValueError("\n".join(lines))
