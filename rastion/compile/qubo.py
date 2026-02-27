"""QUBO conversion helpers."""

from __future__ import annotations

from collections import defaultdict

from rastion.core.ir import QUBOIR, IRObjective
from rastion.core.spec import ObjectiveSense


def objective_to_qubo(objective: IRObjective, n: int) -> QUBOIR:
    q_map: dict[tuple[int, int], float] = defaultdict(float)
    for i, j, v in zip(objective.q_i, objective.q_j, objective.q_v, strict=True):
        a, b = (i, j) if i <= j else (j, i)
        q_map[(a, b)] += float(v)

    pairs = sorted(q_map)
    return QUBOIR(
        n=n,
        q_i=[i for i, _ in pairs],
        q_j=[j for _, j in pairs],
        q_v=[q_map[p] for p in pairs],
        linear=[float(x) for x in objective.linear],
        constant=float(objective.constant),
    )


def qubo_to_objective(qubo: QUBOIR, sense: ObjectiveSense = ObjectiveSense.MIN) -> IRObjective:
    return IRObjective(
        sense=sense,
        linear=[float(x) for x in qubo.linear],
        q_i=list(qubo.q_i),
        q_j=list(qubo.q_j),
        q_v=list(qubo.q_v),
        constant=float(qubo.constant),
    )


def qubo_to_dict(qubo: QUBOIR) -> tuple[dict[tuple[int, int], float], dict[int, float], float]:
    q = {(i, j): float(v) for i, j, v in zip(qubo.q_i, qubo.q_j, qubo.q_v, strict=True)}
    linear = {idx: float(v) for idx, v in enumerate(qubo.linear) if v != 0}
    return q, linear, float(qubo.constant)


def qubo_from_dict(
    n: int,
    q: dict[tuple[int, int], float],
    linear: dict[int, float] | None = None,
    constant: float = 0.0,
) -> QUBOIR:
    linear_values = [0.0] * n
    if linear:
        for idx, value in linear.items():
            linear_values[idx] = float(value)

    q_map: dict[tuple[int, int], float] = defaultdict(float)
    for (i, j), value in q.items():
        a, b = (i, j) if i <= j else (j, i)
        q_map[(a, b)] += float(value)

    pairs = sorted(q_map)
    return QUBOIR(
        n=n,
        q_i=[i for i, _ in pairs],
        q_j=[j for _, j in pairs],
        q_v=[q_map[p] for p in pairs],
        linear=linear_values,
        constant=float(constant),
    )
