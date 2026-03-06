from __future__ import annotations

import numpy as np


def cycle_length(distance_matrix: np.ndarray, route: list[int]) -> float:
    if len(route) < 2:
        return 0.0
    total = 0.0
    for idx in range(len(route) - 1):
        total += float(distance_matrix[route[idx], route[idx + 1]])
    return total


def rotate_to_depot(route: list[int], depot: int) -> list[int]:
    if not route:
        return []

    if route[0] == route[-1]:
        base = route[:-1]
    else:
        base = route[:]

    if depot in base:
        start = base.index(depot)
        rotated = base[start:] + base[:start]
    else:
        rotated = [depot] + [node for node in base if node != depot]

    return rotated + [depot]


def nearest_neighbor_route(distance_matrix: np.ndarray, depot: int, seed: int = 0) -> list[int]:
    n = int(distance_matrix.shape[0])
    unvisited = set(range(n))
    unvisited.discard(depot)

    route = [depot]
    current = depot
    rng = np.random.default_rng(seed)

    while unvisited:
        ordered = sorted(unvisited)
        nearest_dist = min(float(distance_matrix[current, node]) for node in ordered)
        nearest_nodes = [node for node in ordered if float(distance_matrix[current, node]) == nearest_dist]
        next_node = int(nearest_nodes[int(rng.integers(0, len(nearest_nodes)))])
        route.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    route.append(depot)
    return route
