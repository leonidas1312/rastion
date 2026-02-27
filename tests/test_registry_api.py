import zipfile
from pathlib import Path

from rastion.registry.loader import AutoSolver, Problem, Solver
from rastion.registry.manager import init_registry, install_solver_from_url, list_problems
from rastion.solvers.discovery import discover_plugins


def test_registry_initialization_and_problem_loader(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "rastion-home"
    monkeypatch.setenv("RASTION_HOME", str(home))

    init_registry()
    problems = list_problems()

    assert any(entry.name == "knapsack" for entry in problems)

    problem = Problem.from_registry("knapsack")
    assert problem.spec.name == "knapsack"
    assert "default" in problem.instances
    assert problem.metadata["version"] == "0.1.0"
    assert "#" in problem.card

    instance = problem.load_instance("default")
    assert "profits" in instance.arrays


def test_solver_loader_and_auto_solver(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "rastion-home"
    monkeypatch.setenv("RASTION_HOME", str(home))

    init_registry()
    available = Solver.available()

    assert "baseline" in available

    baseline = Solver.from_name("baseline")
    assert baseline.plugin.name == "baseline"

    highs = Solver.from_name("highs")
    assert highs.plugin.name == "highs"

    selected = AutoSolver.from_problem(Problem.from_registry("knapsack"))
    assert selected.plugin.name in set(available)


def test_install_solver_from_url_and_discovery(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "rastion-home"
    monkeypatch.setenv("RASTION_HOME", str(home))
    init_registry()

    archive = tmp_path / "solver.zip"
    solver_source = """
from rastion.core.solution import Solution, SolutionStatus
from rastion.core.spec import VariableType
from rastion.solvers.base import CapabilitySet, ConstraintType, ObjectiveType, SolveMode, SolverPlugin

class ZipTestSolver(SolverPlugin):
    name = "zip_test_solver"
    version = "0.0.1"

    def capabilities(self):
        return CapabilitySet(
            variable_types={VariableType.BINARY},
            objective_types={ObjectiveType.LINEAR},
            constraint_types={ConstraintType.LINEAR},
            modes={SolveMode.SOLVE},
        )

    def solve(self, ir_model, config, backend):
        return Solution(
            status=SolutionStatus.UNKNOWN,
            objective_value=None,
            primal_values={},
            metadata={
                "runtime_s": 0.0,
                "solver_name": self.name,
                "solver_version": self.version,
            },
        )

def get_plugin():
    return ZipTestSolver()
""".strip()

    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("repo-main/solver.py", solver_source)

    installed = install_solver_from_url(archive.resolve().as_uri())
    assert (installed / "solver.py").exists()

    plugins = discover_plugins()
    assert "zip_test_solver" in plugins
