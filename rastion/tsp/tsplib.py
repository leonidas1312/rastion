from __future__ import annotations

from pathlib import Path

import numpy as np

from rastion.tsp.types import TSPProblem


DEFAULT_TSPLIB_INSTANCES: tuple[tuple[str, str], ...] = (
    ("small", "berlin52.tsp"),
    ("medium", "ch150.tsp"),
    ("large", "a280.tsp"),
)


def _parse_header_line(line: str) -> tuple[str, str] | None:
    if ":" not in line:
        return None
    key, value = line.split(":", maxsplit=1)
    return key.strip().upper(), value.strip()


def load_tsplib_problem(path: str | Path) -> TSPProblem:
    tsp_path = Path(path)
    lines = tsp_path.read_text(encoding="utf-8").splitlines()

    header: dict[str, str] = {}
    coords: list[tuple[int, float, float]] = []
    in_coords = False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper == "NODE_COORD_SECTION":
            in_coords = True
            continue
        if upper == "EOF":
            break

        if not in_coords:
            parsed = _parse_header_line(line)
            if parsed is not None:
                key, value = parsed
                header[key] = value
            continue

        parts = line.split()
        if len(parts) < 3:
            continue
        idx = int(parts[0]) - 1
        x = float(parts[1])
        y = float(parts[2])
        coords.append((idx, x, y))

    if not coords:
        raise ValueError(f"No NODE_COORD_SECTION data found in {tsp_path}")

    dimension = int(header.get("DIMENSION", len(coords)))
    edge_weight_type = header.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper().strip()
    if edge_weight_type != "EUC_2D":
        raise ValueError(f"Only EUC_2D TSPLIB problems are supported, got: {edge_weight_type}")

    coord_array = np.zeros((dimension, 2), dtype=float)
    for idx, x, y in coords:
        if idx < 0 or idx >= dimension:
            raise ValueError(f"Coordinate index {idx+1} out of dimension range {dimension}")
        coord_array[idx, 0] = x
        coord_array[idx, 1] = y

    name = header.get("NAME", tsp_path.stem)
    return TSPProblem(
        name=name,
        n_vars=dimension,
        depot=0,
        coords=coord_array,
        edge_weight_type=edge_weight_type,
        source=str(tsp_path),
    )


def default_tsplib_paths(base_dir: str | Path = "examples/tsplib") -> list[tuple[str, Path]]:
    base = Path(base_dir)
    return [(size, base / filename) for size, filename in DEFAULT_TSPLIB_INSTANCES]
