"""Deterministic ASE reading, writing, and property extraction."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write

from molsim_agent.formats.detection import detect_format, normalize_format


def read_structure(path: Path) -> tuple[Atoms, str, list[str]]:
    format_name, warnings = detect_format(path)
    if format_name is None:
        raise ValueError(f"Could not determine a supported structure format for {path.name}")
    atoms = read(path, format=format_name)
    if not isinstance(atoms, Atoms):
        raise ValueError(f"Expected one structure in {path.name}, got a trajectory")
    return atoms, format_name, warnings


def write_structure(path: Path, atoms: Atoms, format_name: str) -> None:
    format_name = normalize_format(format_name)
    options: dict[str, Any] = {}
    if format_name == "lammps-data":
        options = {"atom_style": "atomic", "masses": True}
    write(path, atoms, format=format_name, **options)


def structure_summary(atoms: Atoms) -> dict[str, Any]:
    symbols = atoms.get_chemical_symbols()
    arrays = sorted(atoms.arrays)
    calculator_results = sorted(atoms.calc.results) if atoms.calc is not None else []
    return {
        "atom_count": len(atoms),
        "chemical_symbols": symbols,
        "species_counts": dict(sorted(Counter(symbols).items())),
        "positions_angstrom": atoms.get_positions().tolist(),
        "cell_angstrom": atoms.cell.array.tolist(),
        "pbc": [bool(value) for value in atoms.pbc],
        "arrays": arrays,
        "calculator_properties": calculator_results,
        "velocities_present": atoms.has("momenta"),
        "forces_present": _forces(atoms) is not None,
        "energy_present": _energy(atoms) is not None,
        "charges_present": atoms.has("initial_charges") or _charges(atoms) is not None,
        "constraints": constraint_signatures(atoms),
    }


def constraint_signatures(atoms: Atoms) -> list[dict[str, Any]]:
    signatures = []
    for constraint in atoms.constraints:
        data = constraint.todict() if hasattr(constraint, "todict") else {}
        signatures.append({"type": type(constraint).__name__, "data": _jsonable(data)})
    return signatures


def optional_properties(atoms: Atoms) -> dict[str, Any]:
    return {
        "velocities": atoms.get_velocities() if atoms.has("momenta") else None,
        "constraints": constraint_signatures(atoms) or None,
        "charges": _charges(atoms),
        "forces": _forces(atoms),
        "energy": _energy(atoms),
    }


def _forces(atoms: Atoms) -> np.ndarray | None:
    if atoms.calc is not None and "forces" in atoms.calc.results:
        return np.asarray(atoms.calc.results["forces"])
    if "forces" in atoms.arrays:
        return np.asarray(atoms.arrays["forces"])
    return None


def _energy(atoms: Atoms) -> float | None:
    if atoms.calc is not None and "energy" in atoms.calc.results:
        return float(atoms.calc.results["energy"])
    return None


def _charges(atoms: Atoms) -> np.ndarray | None:
    if atoms.has("initial_charges"):
        return atoms.get_initial_charges()
    if atoms.calc is not None and "charges" in atoms.calc.results:
        return np.asarray(atoms.calc.results["charges"])
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
