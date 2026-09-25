from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read


def trajectory_summary(path: str | Path) -> dict[str, Any]:
    frames = read(path, index=":")
    if not frames:
        raise ValueError("trajectory contains no frames")
    return {"frame_count": len(frames), "atom_count": len(frames[0]), "species": dict(Counter(frames[0].get_chemical_symbols()))}


def pair_distance_statistics(path: str | Path, cutoff_angstrom: float = 10.0) -> dict[str, Any]:
    if cutoff_angstrom <= 0:
        raise ValueError("cutoff_angstrom must be positive")
    frames = read(path, index=":")
    distances = []
    for atoms in frames:
        values = atoms.get_all_distances(mic=bool(atoms.pbc.any()))
        distances.extend(values[np.triu_indices(len(atoms), k=1)].tolist())
    selected = [value for value in distances if value <= cutoff_angstrom]
    return {"frame_count": len(frames), "pair_count": len(selected), "mean_distance_angstrom": float(np.mean(selected)) if selected else None}


def mean_squared_displacement(path: str | Path) -> dict[str, Any]:
    frames = read(path, index=":")
    if len(frames) < 2:
        raise ValueError("MSD requires at least two frames")
    origin = frames[0].get_positions()
    values = [float(np.mean(np.sum((frame.get_positions() - origin) ** 2, axis=1))) for frame in frames]
    return {"frame_count": len(frames), "msd_angstrom2": values}
