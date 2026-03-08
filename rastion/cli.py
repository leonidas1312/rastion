from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from rastion.arena import write_arena_json
from rastion.benchmark import run_benchmark, save_benchmark_results
from rastion.catalog.evals import evaluate_all_suites, write_suite_eval_json
from rastion.catalog.loaders import DEFAULT_CARDS_DIR, DEFAULT_SUITES_DIR, validate_catalog
from rastion.exporters import (
    export_benchmarks_json,
    export_catalog_json,
    export_leaderboards_json,
    export_solvers_json,
    export_suites_json,
)
from rastion.ir.convert import load_problem_json
from rastion.plugins.discovery import discover_solvers
from rastion.solvers.autosolver import AutoSolver
from rastion.solvers.registry import get_solver, list_solvers
from rastion.tsp.arena import write_tsp_arena_bundle

app = typer.Typer(help="Rastion evaluation and publishing CLI")
console = Console()

DEFAULT_WEB_DATA_DIR = Path("web/public/data")
DEFAULT_DEV_EXPORT_DIR = Path(".rastion")
DEFAULT_DEV_WEB_EXPORT_DIR = DEFAULT_DEV_EXPORT_DIR / "exports"
DEFAULT_ARENA_OUT = DEFAULT_DEV_EXPORT_DIR / "arena.json"
DEFAULT_TSP_ARENA_OUT = DEFAULT_WEB_DATA_DIR / "tsp_arena.json"
DEFAULT_BENCHMARKS_EXPORT_OUT = DEFAULT_DEV_WEB_EXPORT_DIR / "benchmarks.json"
DEFAULT_SOLVERS_EXPORT_OUT = DEFAULT_DEV_WEB_EXPORT_DIR / "solvers.json"
DEFAULT_CATALOG_EXPORT_OUT = DEFAULT_WEB_DATA_DIR / "catalog.json"
DEFAULT_SUITES_EXPORT_OUT = DEFAULT_WEB_DATA_DIR / "suites.json"
DEFAULT_LEADERBOARDS_EXPORT_OUT = DEFAULT_WEB_DATA_DIR / "leaderboards.json"
DEFAULT_EVALS_OUT_DIR = DEFAULT_WEB_DATA_DIR / "evals"
DEFAULT_TSPLIB_DIR = Path("examples/tsplib")
DEPRECATED_PUBLIC_ARTIFACTS = (
    DEFAULT_WEB_DATA_DIR / "arena.json",
    DEFAULT_WEB_DATA_DIR / "benchmarks.json",
    DEFAULT_WEB_DATA_DIR / "solvers.json",
)


def _parse_solver_csv(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [name.strip() for name in raw.split(",") if name.strip()]
    return values or None


@app.command("solvers")
def solvers_command() -> None:
    discover_solvers()
    rows = list_solvers()

    table = Table(title="Discovered Solvers")
    table.add_column("Name")
    table.add_column("Supports")
    table.add_column("Max Size", justify="right")
    table.add_column("Hardware")
    table.add_column("Quality", justify="right")
    table.add_column("Source")

    for row in rows:
        solver = row.solver
        meta = row.metadata
        table.add_row(
            row.name,
            ", ".join(solver.supports),
            str(solver.max_size),
            ", ".join(solver.hardware),
            f"{meta.quality:.2f}",
            meta.source,
        )

    console.print(table)


@app.command("solve")
def solve_command(
    problem_path: Path,
    solver: Optional[str] = typer.Option(None, "--solver", help="Use a specific solver name"),
    auto: bool = typer.Option(True, "--auto/--no-auto", help="Use autosolver selection"),
    iters: int = typer.Option(200, "--iters", min=1, help="Iteration budget for toy solvers"),
    seed: int = typer.Option(0, "--seed", help="Deterministic random seed"),
) -> None:
    problem = load_problem_json(problem_path)
    discover_solvers()

    if solver:
        solver_impl = get_solver(solver)
        solution = solver_impl.solve(problem, iters=iters, seed=seed)
        solution.selection_reason = f"selected explicitly via --solver {solver}"
    elif auto:
        solution = AutoSolver().solve(problem, iters=iters, seed=seed)
    else:
        compatible = [r for r in list_solvers() if problem.problem_type in r.solver.supports]
        if not compatible:
            raise typer.BadParameter("No compatible solver available")
        chosen = compatible[0].solver
        solution = chosen.solve(problem, iters=iters, seed=seed)
        solution.selection_reason = f"auto disabled; chose first compatible solver {chosen.name}"

    table = Table(title=f"Solution: {problem.name}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Solver", solution.solver_name)
    table.add_row("Best Value", f"{solution.best_value:.6f}")
    table.add_row("Best x", str(solution.best_x.tolist()))
    if solution.selection_reason:
        table.add_row("Selection", solution.selection_reason)

    console.print(table)


@app.command("benchmark")
def benchmark_command(
    problem_path: Path,
    repeat: int = typer.Option(3, "--repeat", min=1, help="Repetitions per solver"),
    iters: int = typer.Option(200, "--iters", min=1, help="Iteration budget per solve"),
    seed: int = typer.Option(0, "--seed", help="Deterministic random seed"),
) -> None:
    problem = load_problem_json(problem_path)
    discover_solvers()
    rows = run_benchmark(problem, repeat=repeat, iters=iters, seed=seed)
    save_benchmark_results(rows, problem=problem)

    table = Table(title=f"Benchmark: {problem.name}")
    table.add_column("Solver")
    table.add_column("Best Value", justify="right")
    table.add_column("Avg Runtime (s)", justify="right")

    for row in rows:
        table.add_row(
            str(row["solver_name"]),
            f"{float(row['best_value']):.6f}",
            f"{float(row['avg_runtime']):.6f}",
        )

    console.print(table)


@app.command("arena")
def arena_command(
    problem_path: Path,
    out: Path = typer.Option(DEFAULT_ARENA_OUT, "--out", help="Output JSON path"),
    solvers: str | None = typer.Option(None, "--solvers", help="CSV subset, e.g. tabu,simulated_annealing"),
    iters: int = typer.Option(2_000, "--iters", min=1, help="Iteration budget"),
    time_budget_ms: int | None = typer.Option(None, "--time-budget-ms", min=1, help="Wall-clock budget in ms"),
    seed: int = typer.Option(0, "--seed", help="Deterministic random seed"),
    emit_every: int = typer.Option(50, "--emit-every", min=1, help="Emit progress every N iterations"),
) -> None:
    problem = load_problem_json(problem_path)
    discover_solvers()

    solver_names = _parse_solver_csv(solvers)
    payload = write_arena_json(
        problem,
        out,
        solver_names=solver_names,
        iters=iters,
        time_budget_ms=time_budget_ms,
        seed=seed,
        emit_every=emit_every,
    )

    solver_rows = payload.get("solvers", [])
    if not solver_rows:
        raise typer.BadParameter("No compatible solvers selected for this problem")

    table = Table(title=f"Arena: {problem.name}")
    table.add_column("Solver")
    table.add_column("Best Value", justify="right")
    table.add_column("Runtime (ms)", justify="right")
    table.add_column("Events", justify="right")

    for row in solver_rows:
        final = row.get("final", {})
        events = row.get("events", [])
        table.add_row(
            str(row.get("name", "unknown")),
            f"{float(final.get('best_value', 0.0)):.6f}",
            str(int(final.get("runtime_ms", 0))),
            str(len(events) if isinstance(events, list) else 0),
        )

    console.print(table)
    console.print(f"Saved arena payload: {out}")


@app.command("export-benchmarks")
def export_benchmarks_command(
    out: Path = typer.Option(DEFAULT_BENCHMARKS_EXPORT_OUT, "--out", help="Output JSON path for developer exports"),
) -> None:
    payload = export_benchmarks_json(out)
    count = len(payload.get("rows", []))
    console.print(f"Exported {count} benchmark rows to {out}")


@app.command("export-solvers")
def export_solvers_command(
    out: Path = typer.Option(DEFAULT_SOLVERS_EXPORT_OUT, "--out", help="Output JSON path for developer exports"),
) -> None:
    discover_solvers()
    payload = export_solvers_json(out)
    count = int(payload.get("count", 0))
    console.print(f"Exported {count} solvers to {out}")


@app.command("validate-cards")
def validate_cards_command(
    cards_dir: Path = typer.Option(DEFAULT_CARDS_DIR, "--cards-dir", help="Directory with solver card JSON files"),
    suites_dir: Path = typer.Option(DEFAULT_SUITES_DIR, "--suites-dir", help="Directory with suite spec JSON files"),
) -> None:
    cards, suites = validate_catalog(cards_dir=cards_dir, suites_dir=suites_dir)
    console.print(f"Validated {len(cards)} solver cards and {len(suites)} suite specs.")


@app.command("export-catalog")
def export_catalog_command(
    out: Path = typer.Option(DEFAULT_CATALOG_EXPORT_OUT, "--out", help="Output JSON path"),
    cards_dir: Path = typer.Option(DEFAULT_CARDS_DIR, "--cards-dir", help="Directory with solver card JSON files"),
    suites_dir: Path = typer.Option(DEFAULT_SUITES_DIR, "--suites-dir", help="Directory with suite spec JSON files"),
) -> None:
    payload = export_catalog_json(out, cards_dir=cards_dir, suites_dir=suites_dir)
    console.print(f"Exported {int(payload.get('count', 0))} catalog entries to {out}")


@app.command("export-suites")
def export_suites_command(
    out: Path = typer.Option(DEFAULT_SUITES_EXPORT_OUT, "--out", help="Output JSON path"),
    suites_dir: Path = typer.Option(DEFAULT_SUITES_DIR, "--suites-dir", help="Directory with suite spec JSON files"),
) -> None:
    payload = export_suites_json(out, suites_dir=suites_dir)
    console.print(f"Exported {int(payload.get('count', 0))} suite specs to {out}")


@app.command("eval-suite")
def eval_suite_command(
    suite_id: str,
    out: Path | None = typer.Option(None, "--out", help="Output JSON path. Defaults to web/public/data/evals/<suite>.json"),
    cards_dir: Path = typer.Option(DEFAULT_CARDS_DIR, "--cards-dir", help="Directory with solver card JSON files"),
    suites_dir: Path = typer.Option(DEFAULT_SUITES_DIR, "--suites-dir", help="Directory with suite spec JSON files"),
) -> None:
    resolved_out = out or (DEFAULT_EVALS_OUT_DIR / f"{suite_id}.json")
    payload = write_suite_eval_json(resolved_out, suite_id, cards_dir=cards_dir, suites_dir=suites_dir)
    standings = payload.get("standings", [])
    table = Table(title=f"Suite Evaluation: {suite_id}")
    table.add_column("Solver")
    table.add_column("Tier")
    table.add_column("Mean Score", justify="right")
    table.add_column("Mean Runtime (ms)", justify="right")
    table.add_column("Completed", justify="right")
    for row in standings:
        table.add_row(
            str(row.get("solver_name", "?")),
            str(row.get("listing_tier", "?")),
            "n/a" if row.get("mean_score") is None else f"{float(row['mean_score']):.2f}",
            "n/a" if row.get("mean_runtime_ms") is None else f"{float(row['mean_runtime_ms']):.2f}",
            f"{int(row.get('completed_instances', 0))}/{int(row.get('total_instances', 0))}",
        )
    console.print(table)
    console.print(f"Saved suite evaluation: {resolved_out}")


@app.command("export-leaderboards")
def export_leaderboards_command(
    out: Path = typer.Option(DEFAULT_LEADERBOARDS_EXPORT_OUT, "--out", help="Output JSON path"),
    evals_out_dir: Path = typer.Option(DEFAULT_EVALS_OUT_DIR, "--evals-out-dir", help="Directory for per-suite eval JSON"),
    cards_dir: Path = typer.Option(DEFAULT_CARDS_DIR, "--cards-dir", help="Directory with solver card JSON files"),
    suites_dir: Path = typer.Option(DEFAULT_SUITES_DIR, "--suites-dir", help="Directory with suite spec JSON files"),
) -> None:
    payloads = evaluate_all_suites(out_dir=evals_out_dir, cards_dir=cards_dir, suites_dir=suites_dir)
    payload = export_leaderboards_json(
        out,
        suite_payloads=payloads,
        evals_out_dir=evals_out_dir,
        cards_dir=cards_dir,
        suites_dir=suites_dir,
    )
    console.print(f"Exported {int(payload.get('count', 0))} result bundles to {out}")


@app.command("tsp-arena")
def tsp_arena_command(
    out: Path = typer.Option(DEFAULT_TSP_ARENA_OUT, "--out", help="Output JSON path"),
    tsplib_dir: Path = typer.Option(DEFAULT_TSPLIB_DIR, "--tsplib-dir", help="Directory with TSPLIB .tsp files"),
    solvers: str | None = typer.Option(None, "--solvers", help="CSV subset, e.g. tsp_nearest_neighbor,tsp_two_opt"),
    iters: int = typer.Option(2_000, "--iters", min=1, help="Iteration budget for iterative heuristics"),
    time_budget_ms: int | None = typer.Option(None, "--time-budget-ms", min=1, help="Wall-clock budget in ms"),
    seed: int = typer.Option(0, "--seed", help="Deterministic random seed"),
    emit_every: int = typer.Option(50, "--emit-every", min=1, help="Emit progress every N iterations"),
) -> None:
    discover_solvers()
    solver_names = _parse_solver_csv(solvers)
    payload = write_tsp_arena_bundle(
        out,
        tsplib_dir=tsplib_dir,
        solver_names=solver_names,
        iters=iters,
        time_budget_ms=time_budget_ms,
        seed=seed,
        emit_every=emit_every,
    )

    instances = payload.get("instances", [])
    table = Table(title="TSP Arena Bundle")
    table.add_column("Size")
    table.add_column("Instance")
    table.add_column("n")
    table.add_column("Solvers")
    for instance in instances:
        solver_count = len(instance.get("solvers", [])) if isinstance(instance, dict) else 0
        table.add_row(
            str(instance.get("size", "?")),
            str(instance.get("name", "?")),
            str(instance.get("n_vars", "?")),
            str(solver_count),
        )
    console.print(table)
    console.print(f"Saved TSPLIB arena payload: {out}")


@app.command("build-site-data")
def build_site_data_command(
    iters: int = typer.Option(1_000, "--iters", min=1, help="Arena iteration budget"),
    seed: int = typer.Option(0, "--seed", help="Deterministic random seed"),
    emit_every: int = typer.Option(50, "--emit-every", min=1, help="Arena event cadence"),
    cards_dir: Path = typer.Option(DEFAULT_CARDS_DIR, "--cards-dir", help="Directory with solver card JSON files"),
    suites_dir: Path = typer.Option(DEFAULT_SUITES_DIR, "--suites-dir", help="Directory with suite spec JSON files"),
) -> None:
    validate_catalog(cards_dir=cards_dir, suites_dir=suites_dir)
    discover_solvers()

    for stale_path in DEPRECATED_PUBLIC_ARTIFACTS:
        if stale_path.exists():
            stale_path.unlink()

    suite_payloads = evaluate_all_suites(
        out_dir=DEFAULT_EVALS_OUT_DIR,
        cards_dir=cards_dir,
        suites_dir=suites_dir,
    )
    export_catalog_json(DEFAULT_CATALOG_EXPORT_OUT, cards_dir=cards_dir, suites_dir=suites_dir)
    export_suites_json(DEFAULT_SUITES_EXPORT_OUT, suites_dir=suites_dir)
    export_leaderboards_json(
        DEFAULT_LEADERBOARDS_EXPORT_OUT,
        suite_payloads=suite_payloads,
        evals_out_dir=DEFAULT_EVALS_OUT_DIR,
        cards_dir=cards_dir,
        suites_dir=suites_dir,
    )
    write_tsp_arena_bundle(
        DEFAULT_TSP_ARENA_OUT,
        tsplib_dir=DEFAULT_TSPLIB_DIR,
        iters=iters,
        seed=seed,
        emit_every=emit_every,
    )

    console.print("Public TSP hub data generated for website:")
    console.print(f"  - {DEFAULT_CATALOG_EXPORT_OUT}")
    console.print(f"  - {DEFAULT_SUITES_EXPORT_OUT}")
    console.print(f"  - {DEFAULT_LEADERBOARDS_EXPORT_OUT}")
    console.print(f"  - {DEFAULT_EVALS_OUT_DIR}")
    console.print(f"  - {DEFAULT_TSP_ARENA_OUT}")
    console.print("Run locally:")
    console.print("  cd web && npm install && npm run dev")


@app.command("demo-site")
def demo_site_command(
    iters: int = typer.Option(1_000, "--iters", min=1, help="Arena iteration budget"),
    seed: int = typer.Option(0, "--seed", help="Deterministic random seed"),
    emit_every: int = typer.Option(50, "--emit-every", min=1, help="Arena event cadence"),
) -> None:
    console.print("[yellow]demo-site is retained as an alias. Use build-site-data for the public publication flow.[/yellow]")
    build_site_data_command(iters=iters, seed=seed, emit_every=emit_every)


@app.command("demo")
def demo_command(
    iters: int = typer.Option(1_000, "--iters", min=1, help="Arena iteration budget"),
    seed: int = typer.Option(0, "--seed", help="Deterministic random seed"),
    emit_every: int = typer.Option(50, "--emit-every", min=1, help="Arena event cadence"),
) -> None:
    build_site_data_command(iters=iters, seed=seed, emit_every=emit_every)


if __name__ == "__main__":
    app()
