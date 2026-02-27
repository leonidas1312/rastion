"""Qiskit QAOA solver plugin for small QUBO problems."""

from __future__ import annotations

import time
from collections.abc import Mapping

from rastion.core.ir import QUBOIR, IRModel
from rastion.core.solution import Solution, SolutionStatus
from rastion.core.spec import ObjectiveSense, VariableType
from rastion.solvers.base import CapabilitySet, ObjectiveType, SolveMode, SolverPlugin


class QAOAPlugin(SolverPlugin):
    """QUBO sampler based on Qiskit QAOA."""

    name = "qaoa"

    def __init__(self, version: str) -> None:
        self.version = version

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            variable_types={VariableType.BINARY},
            objective_types={ObjectiveType.QUBO},
            constraint_types=set(),
            modes={SolveMode.SAMPLE},
            result_fields={"objective_value", "primal_values", "runtime_s"},
        )

    def solve(self, ir_model: IRModel, config: dict[str, object], backend: object) -> Solution:
        if ir_model.qubo is None:
            raise ValueError("qaoa plugin requires QUBO IR")
        if any(var.vartype != VariableType.BINARY for var in ir_model.variables):
            raise ValueError("qaoa plugin supports binary variables only")
        if ir_model.constraints is not None and ir_model.constraints.num_rows > 0:
            raise ValueError("qaoa plugin does not support linear constraints")

        if ir_model.objective.sense == ObjectiveSense.MAX:
            raise ValueError("qaoa plugin expects minimization QUBO objective")

        try:
            try:
                from qiskit.primitives import Sampler as PrimitiveSampler
            except ImportError:
                from qiskit.primitives import StatevectorSampler as PrimitiveSampler
            from qiskit_algorithms.minimum_eigensolvers import QAOA
            from qiskit_algorithms.optimizers import COBYLA, SLSQP, SPSA
            from qiskit_algorithms.utils import algorithm_globals
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "QAOA solver requires qiskit and qiskit-algorithms extras"
            ) from exc

        reps = int(config.get("reps", 2))
        optimizer_name = str(config.get("optimizer", "COBYLA")).upper()
        num_samples = int(config.get("num_samples", 1000))
        seed = config.get("seed")

        if seed is not None:
            algorithm_globals.random_seed = int(seed)

        optimizer = self._build_optimizer(optimizer_name, COBYLA, SPSA, SLSQP)
        sampler = self._build_sampler(PrimitiveSampler, num_samples=num_samples, seed=seed)

        operator = self._qubo_to_sparse_pauli(ir_model.qubo)

        start = time.perf_counter()
        result = QAOA(sampler=sampler, optimizer=optimizer, reps=reps).compute_minimum_eigenvalue(operator)
        runtime_s = time.perf_counter() - start

        bitstrings = self._extract_bitstrings(result, ir_model.qubo.n)
        if not bitstrings:
            raise RuntimeError("QAOA returned no samples")

        best_x: list[int] | None = None
        best_obj: float | None = None
        for candidate in bitstrings:
            value = self._qubo_value(ir_model.qubo, candidate)
            if best_obj is None or value < best_obj - 1e-12:
                best_x = candidate
                best_obj = value

        assert best_x is not None and best_obj is not None

        primal_values = {
            ir_model.variables[idx].name: float(best_x[idx])
            for idx in range(len(ir_model.variables))
        }

        return Solution(
            status=SolutionStatus.FEASIBLE,
            objective_value=float(best_obj),
            primal_values=primal_values,
            metadata={
                "runtime_s": runtime_s,
                "solver_name": self.name,
                "solver_version": self.version,
                "reps": reps,
                "optimizer": optimizer_name,
                "num_samples": num_samples,
            },
        )

    def _build_optimizer(self, name: str, cobyla: object, spsa: object, slsqp: object) -> object:
        if name == "COBYLA":
            return cobyla(maxiter=250)
        if name == "SPSA":
            return spsa(maxiter=250)
        if name == "SLSQP":
            return slsqp(maxiter=250)
        raise ValueError("qaoa optimizer must be one of: COBYLA, SPSA, SLSQP")

    def _build_sampler(self, sampler_cls: object, num_samples: int, seed: object) -> object:
        if getattr(sampler_cls, "__name__", "") == "StatevectorSampler":
            kwargs: dict[str, object] = {"default_shots": num_samples}
            if seed is not None:
                kwargs["seed"] = int(seed)
            return sampler_cls(**kwargs)

        try:
            options: dict[str, object] = {"shots": num_samples}
            if seed is not None:
                options["seed"] = int(seed)
            return sampler_cls(options=options)
        except TypeError:
            sampler = sampler_cls()
            if hasattr(sampler, "options"):
                if hasattr(sampler.options, "shots"):
                    sampler.options.shots = num_samples
                if seed is not None and hasattr(sampler.options, "seed"):
                    sampler.options.seed = int(seed)
            return sampler

    def _qubo_to_sparse_pauli(self, qubo: QUBOIR) -> object:
        from qiskit.quantum_info import SparsePauliOp

        n = qubo.n
        constant = float(qubo.constant)
        h: list[float] = [0.0] * n
        j_terms: dict[tuple[int, int], float] = {}

        for idx, coeff in enumerate(qubo.linear):
            c = float(coeff)
            if c == 0.0:
                continue
            constant += 0.5 * c
            h[idx] += -0.5 * c

        for i, j, coeff in zip(qubo.q_i, qubo.q_j, qubo.q_v, strict=True):
            q = float(coeff)
            if q == 0.0:
                continue
            if i == j:
                constant += 0.5 * q
                h[i] += -0.5 * q
            else:
                a, b = (i, j) if i < j else (j, i)
                constant += 0.25 * q
                h[a] += -0.25 * q
                h[b] += -0.25 * q
                j_terms[(a, b)] = j_terms.get((a, b), 0.0) + 0.25 * q

        paulis: list[tuple[str, float]] = []
        if constant != 0.0:
            paulis.append(("I" * n, constant))

        for idx, coeff in enumerate(h):
            if coeff == 0.0:
                continue
            paulis.append((self._z_string(n, idx), coeff))

        for (i, j), coeff in j_terms.items():
            if coeff == 0.0:
                continue
            paulis.append((self._zz_string(n, i, j), coeff))

        if not paulis:
            paulis.append(("I" * n, 0.0))

        return SparsePauliOp.from_list(paulis)

    def _z_string(self, n: int, idx: int) -> str:
        chars = ["I"] * n
        chars[n - 1 - idx] = "Z"
        return "".join(chars)

    def _zz_string(self, n: int, i: int, j: int) -> str:
        chars = ["I"] * n
        chars[n - 1 - i] = "Z"
        chars[n - 1 - j] = "Z"
        return "".join(chars)

    def _extract_bitstrings(self, result: object, n: int) -> list[list[int]]:
        candidates: list[list[int]] = []

        best_measurement = getattr(result, "best_measurement", None)
        if isinstance(best_measurement, Mapping):
            bitstring = best_measurement.get("bitstring")
            if isinstance(bitstring, str):
                candidates.append(self._bitstring_to_vector(bitstring, n))

        eigenstate = getattr(result, "eigenstate", None)
        if hasattr(eigenstate, "binary_probabilities"):
            probs = eigenstate.binary_probabilities()
            if isinstance(probs, Mapping):
                for key in sorted(probs, key=lambda k: probs[k], reverse=True):
                    candidates.append(self._bitstring_to_vector(str(key), n))
        elif isinstance(eigenstate, Mapping):
            for key in sorted(eigenstate, key=lambda k: eigenstate[k], reverse=True):
                candidates.append(self._key_to_vector(key, n))

        # Preserve order and deduplicate.
        unique: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for item in candidates:
            key = tuple(item)
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    def _key_to_vector(self, key: object, n: int) -> list[int]:
        if isinstance(key, int):
            return self._bitstring_to_vector(format(key, f"0{n}b"), n)
        return self._bitstring_to_vector(str(key), n)

    def _bitstring_to_vector(self, bitstring: str, n: int) -> list[int]:
        bits = bitstring.replace(" ", "").zfill(n)
        bits = bits[-n:]
        # Qiskit bitstrings are big-endian in text form; reverse for variable index order.
        return [int(ch) for ch in bits[::-1]]

    def _qubo_value(self, qubo: QUBOIR, x: list[int]) -> float:
        value = float(qubo.constant)
        value += sum(float(c) * x[i] for i, c in enumerate(qubo.linear))
        for i, j, coeff in zip(qubo.q_i, qubo.q_j, qubo.q_v, strict=True):
            value += float(coeff) * x[i] * x[j]
        return value


def get_plugin() -> QAOAPlugin | None:
    try:
        import qiskit
        import qiskit_algorithms  # noqa: F401
    except ImportError:
        return None
    version = getattr(qiskit, "__version__", "unknown")
    return QAOAPlugin(version=version)
