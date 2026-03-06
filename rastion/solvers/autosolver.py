from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rastion.ir.types import ProblemIR
from rastion.solvers.registry import RegisteredSolver, list_solvers
from rastion.solvers.base import Solution


DEFAULT_BENCHMARK_PATH = Path(".rastion/benchmarks.json")


class AutoSolver:
    def __init__(
        self,
        *,
        w1: float = 0.4,
        w2: float = 0.4,
        w3: float = 0.2,
        benchmark_cache_path: str | Path = DEFAULT_BENCHMARK_PATH,
    ) -> None:
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.benchmark_cache_path = Path(benchmark_cache_path)

    def _load_benchmark_hint(self, problem_type: str, n_vars: int, solver_name: str) -> float | None:
        if not self.benchmark_cache_path.exists():
            return None
        try:
            payload = json.loads(self.benchmark_cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

        key = f"{problem_type}|{n_vars}|{solver_name}"
        row = payload.get("entries", {}).get(key)
        if not isinstance(row, dict):
            return None
        hint = row.get("hint")
        if hint is None:
            return None
        return float(min(max(hint, 0.0), 1.0))

    def _score(self, problem: ProblemIR, reg: RegisteredSolver) -> tuple[float, str]:
        solver = reg.solver
        metadata = reg.metadata

        size_fit = 1.0 - (problem.n_vars / max(solver.max_size, 1))
        size_fit = float(min(max(size_fit, 0.0), 1.0))
        preference = float(min(max(metadata.quality, 0.0), 1.0))
        benchmark_hint = self._load_benchmark_hint(problem.problem_type, problem.n_vars, solver.name)
        benchmark_component = benchmark_hint if benchmark_hint is not None else 0.0

        score = self.w1 * size_fit + self.w2 * preference + self.w3 * benchmark_component
        reason = (
            f"size_fit={size_fit:.3f}, quality={preference:.3f}, "
            f"benchmark_hint={(benchmark_hint if benchmark_hint is not None else 'missing')}"
        )
        return score, reason

    def choose_solver(self, problem: ProblemIR) -> list[tuple[RegisteredSolver, float, str]]:
        candidates = []
        for reg in list_solvers():
            solver = reg.solver
            if problem.problem_type not in solver.supports:
                continue
            if problem.n_vars > solver.max_size:
                continue
            score, reason = self._score(problem, reg)
            candidates.append((reg, score, reason))

        return sorted(candidates, key=lambda item: item[1], reverse=True)

    def solve(self, problem: ProblemIR, **kwargs: Any) -> Solution:
        ranked = self.choose_solver(problem)
        if not ranked:
            raise RuntimeError(f"No compatible solver found for {problem.problem_type} with n={problem.n_vars}")

        failures: list[str] = []
        for reg, score, reason in ranked:
            solver = reg.solver
            try:
                solution = solver.solve(problem, **kwargs)
                suffix = f"selected={solver.name}, score={score:.3f}, details=({reason})"
                if failures:
                    suffix += f", fallbacks={'; '.join(failures)}"
                solution.selection_reason = suffix
                return solution
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{solver.name} failed: {exc}")

        raise RuntimeError("All candidate solvers failed: " + "; ".join(failures))
