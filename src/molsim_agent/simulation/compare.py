"""Comparisons on identical fixed configurations (not frame-by-frame MD matching)."""
from __future__ import annotations

from typing import Any

import numpy as np
from molsim_agent.formats.structures import read_structure
from molsim_agent.potentials.registry import PotentialRegistry
from molsim_agent.safety.policies import Workspace


def compare_single_point(workspace: Workspace, structure: str, calculators: list[str]) -> dict[str, Any]:
    if len(calculators) < 2:
        raise ValueError("compare_single_point requires at least two calculators")
    source = workspace.resolve(structure, must_exist=True)
    atoms, source_format, warnings = read_structure(source)
    registry = PotentialRegistry()
    results: dict[str, Any] = {}
    energies: dict[str, float] = {}
    forces: dict[str, np.ndarray] = {}
    for name in calculators:
        trial = atoms.copy()
        trial.calc = registry.create_calculator(name)
        energies[name] = float(trial.get_potential_energy())
        forces[name] = np.asarray(trial.get_forces())
        results[name] = {"energy_eV": energies[name], "forces_eV_per_angstrom": forces[name].tolist()}
    reference = calculators[0]
    results["differences_to_reference"] = {
        name: {
            "energy_delta_eV": energies[name] - energies[reference],
            "max_force_delta_eV_per_angstrom": float(np.max(np.abs(forces[name] - forces[reference]))),
        }
        for name in calculators[1:]
    }
    return {"ok": True, "structure": workspace.relative(source), "source_format": source_format,
            "atom_count": len(atoms), "results": results, "warnings": warnings}
