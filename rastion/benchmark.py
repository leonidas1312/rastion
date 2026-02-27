"""Benchmark utilities for comparing solver performance."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

from rastion.backends.local import LocalBackend
from rastion.compile.normalize import compile_to_ir
from rastion.core.data import InstanceData
from rastion.core.run_record import append_run_record, create_run_record
from rastion.core.solution import Solution, SolutionStatus
from rastion.registry.loader import Problem
from rastion.registry.manager import init_registry, runs_root
from rastion.solvers.discovery import discover_plugins


@dataclass
class BenchmarkResult:
    solver: str
    runtime: float
    objective: float | None
    status: str
    gap: float | None
    runs: int
    succeeded_runs: int
    errors: list[str]


@dataclass
class BenchmarkSuiteRow:
    problem: str
    instance: str
    solver: str
    runtime: float
    objective: float | None
    status: str
    gap: float | None


class BenchmarkSuiteResults:
    """Collection returned by `run_benchmark_suite`."""

    def __init__(self, rows: list[BenchmarkSuiteRow]) -> None:
        self.rows = rows

    def to_csv(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["problem", "instance", "solver", "runtime", "objective", "status", "gap"],
            )
            writer.writeheader()
            for row in self.rows:
                writer.writerow(
                    {
                        "problem": row.problem,
                        "instance": row.instance,
                        "solver": row.solver,
                        "runtime": row.runtime,
                        "objective": row.objective,
                        "status": row.status,
                        "gap": row.gap,
                    }
                )
        return target

    def to_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "problem": row.problem,
                "instance": row.instance,
                "solver": row.solver,
                "runtime": row.runtime,
                "objective": row.objective,
                "status": row.status,
                "gap": row.gap,
            }
            for row in self.rows
        ]
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target


def compare(
    problem: Problem | str,
    instance: str = "default",
    solvers: list[str] | None = None,
    time_limit: str | float | None = "30s",
    runs: int = 1,
    save_to_history: bool = True,
) -> list[BenchmarkResult]:
    """Compare multiple solvers on one problem instance."""
    problem_obj = _resolve_problem(problem)
    instance_data = problem_obj.load_instance(instance)
    ir_model = compile_to_ir(problem_obj.spec, instance_data)

    plugins = discover_plugins()
    if not plugins:
        raise ValueError("no solver plugins available")

    solver_names = solvers or sorted(plugins)
    backend = LocalBackend()
    time_limit_seconds = _parse_time_limit(time_limit)

    results: list[BenchmarkResult] = []
    for solver_name in solver_names:
        plugin = plugins.get(solver_name)
        if plugin is None:
            results.append(
                BenchmarkResult(
                    solver=solver_name,
                    runtime=0.0,
                    objective=None,
                    status=SolutionStatus.ERROR.value,
                    gap=None,
                    runs=runs,
                    succeeded_runs=0,
                    errors=[f"solver '{solver_name}' is not installed"],
                )
            )
            continue

        runtimes: list[float] = []
        objectives: list[float] = []
        gaps: list[float] = []
        statuses: list[SolutionStatus] = []
        errors: list[str] = []

        for _ in range(max(1, runs)):
            config: dict[str, object] = {}
            if time_limit_seconds is not None:
                config["time_limit"] = time_limit_seconds

            start = time.perf_counter()
            try:
                solution = backend.run(plugin, ir_model, config)
            except Exception as exc:  # pragma: no cover - optional solver runtime behavior
                runtime = time.perf_counter() - start
                runtimes.append(runtime)
                statuses.append(SolutionStatus.ERROR)
                errors.append(str(exc))
                if save_to_history:
                    _save_benchmark_run(
                        problem=problem_obj,
                        instance=instance_data,
                        solver_name=plugin.name,
                        solver_version=plugin.version,
                        solution=Solution(
                            status=SolutionStatus.ERROR,
                            objective_value=None,
                            primal_values={},
                            metadata={
                                "runtime_s": runtime,
                                "solver_name": plugin.name,
                                "solver_version": plugin.version,
                            },
                            error_message=str(exc),
                        ),
                        solver_config=config,
                    )
                continue

            runtime = float(solution.metadata.get("runtime_s", time.perf_counter() - start))
            runtimes.append(runtime)
            statuses.append(solution.status)

            if solution.objective_value is not None:
                objectives.append(float(solution.objective_value))

            gap = solution.metadata.get("gap")
            if gap is not None:
                try:
                    gaps.append(float(gap))
                except (TypeError, ValueError):
                    pass

            if solution.error_message:
                errors.append(solution.error_message)

            if save_to_history:
                _save_benchmark_run(
                    problem=problem_obj,
                    instance=instance_data,
                    solver_name=plugin.name,
                    solver_version=plugin.version,
                    solution=solution,
                    solver_config=config,
                )

        success_count = sum(1 for status in statuses if status in {SolutionStatus.OPTIMAL, SolutionStatus.FEASIBLE})
        avg_runtime = sum(runtimes) / len(runtimes) if runtimes else 0.0
        avg_gap = (sum(gaps) / len(gaps)) if gaps else None
        objective = _select_objective(objectives, statuses)
        summary_status = _summarize_status(statuses)

        results.append(
            BenchmarkResult(
                solver=solver_name,
                runtime=avg_runtime,
                objective=objective,
                status=summary_status,
                gap=avg_gap,
                runs=max(1, runs),
                succeeded_runs=success_count,
                errors=errors,
            )
        )

    return results


def _save_benchmark_run(
    *,
    problem: Problem,
    instance: InstanceData,
    solver_name: str,
    solver_version: str,
    solution: Solution,
    solver_config: dict[str, object],
) -> None:
    """Persist a benchmark solver run to run-history JSONL."""
    run_record = create_run_record(
        spec=problem.spec,
        instance=instance,
        solution=solution,
        solver_name=solver_name,
        solver_version=solver_version,
        solver_config=solver_config,
    )
    append_run_record(run_record, runs_dir=runs_root())


def run_benchmark_suite(
    problems: list[str],
    solvers: list[str],
    *,
    instance: str = "default",
    time_limit: str | float | None = "60s",
    runs: int = 1,
) -> BenchmarkSuiteResults:
    """Run solver benchmarks on multiple problems."""
    init_registry()

    rows: list[BenchmarkSuiteRow] = []
    for problem_name in problems:
        problem = Problem.from_registry(problem_name)
        comparison = compare(
            problem,
            instance=instance,
            solvers=solvers,
            time_limit=time_limit,
            runs=runs,
        )
        for result in comparison:
            rows.append(
                BenchmarkSuiteRow(
                    problem=problem_name,
                    instance=instance,
                    solver=result.solver,
                    runtime=result.runtime,
                    objective=result.objective,
                    status=result.status,
                    gap=result.gap,
                )
            )

    return BenchmarkSuiteResults(rows)


def _resolve_problem(problem: Problem | str) -> Problem:
    if isinstance(problem, Problem):
        return problem

    text = str(problem)
    candidate = Path(text).expanduser().resolve()
    if candidate.exists():
        return Problem.from_local(candidate)

    init_registry()
    return Problem.from_registry(text)


def _parse_time_limit(value: str | float | None) -> float | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = value.strip().lower()
    if not text:
        return None

    if text.endswith("ms"):
        return float(text[:-2]) / 1000.0
    if text.endswith("s"):
        return float(text[:-1])
    if text.endswith("m"):
        return float(text[:-1]) * 60.0
    return float(text)


def _summarize_status(statuses: list[SolutionStatus]) -> str:
    if not statuses:
        return SolutionStatus.UNKNOWN.value
    if any(status == SolutionStatus.OPTIMAL for status in statuses):
        return SolutionStatus.OPTIMAL.value
    if any(status == SolutionStatus.FEASIBLE for status in statuses):
        return SolutionStatus.FEASIBLE.value
    if any(status == SolutionStatus.INFEASIBLE for status in statuses):
        return SolutionStatus.INFEASIBLE.value
    if any(status == SolutionStatus.ERROR for status in statuses):
        return SolutionStatus.ERROR.value
    return SolutionStatus.UNKNOWN.value


def _select_objective(objectives: list[float], statuses: list[SolutionStatus]) -> float | None:
    if not objectives:
        return None

    if any(status == SolutionStatus.OPTIMAL for status in statuses):
        return min(objectives)
    return min(objectives)
