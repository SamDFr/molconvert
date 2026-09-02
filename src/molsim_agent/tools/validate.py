"""Scientific comparison of source and converted ASE structures."""

from __future__ import annotations

from typing import Any

import numpy as np

from molsim_agent.formats.structures import optional_properties, read_structure
from molsim_agent.safety.policies import Workspace
from molsim_agent.tools.registry import ToolSpec


def validate_conversion(
    workspace: Workspace,
    source: str,
    destination: str,
    tolerance: float = 1e-8,
) -> dict[str, object]:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    source_path = workspace.resolve(source, must_exist=True)
    destination_path = workspace.resolve(destination, must_exist=True)
    before, source_format, source_warnings = read_structure(source_path)
    after, destination_format, destination_warnings = read_structure(destination_path)
    warnings = [*source_warnings, *destination_warnings]

    atom_count = "preserved" if len(before) == len(after) else "changed"
    before_symbols = before.get_chemical_symbols()
    after_symbols = after.get_chemical_symbols()
    species = "preserved" if before_symbols == after_symbols else "changed"
    positions = _array_comparison(before.positions, after.positions, tolerance, "angstrom")
    cell = _array_comparison(before.cell.array, after.cell.array, tolerance, "angstrom")
    if destination_format == "xyz" and bool(before.cell.any()):
        cell = {"status": "lost", "reason": "plain_xyz_does_not_encode_cell"}
    if destination_format == "xyz" and bool(before.pbc.any()):
        pbc = "lost"
    elif destination_format == "lammps-data":
        pbc = "not_encoded"
    else:
        pbc = "preserved" if np.array_equal(before.pbc, after.pbc) else "changed"

    before_optional = optional_properties(before)
    after_optional = optional_properties(after)
    optional = {
        name: _optional_comparison(before_optional[name], after_optional[name], tolerance)
        for name in ("velocities", "constraints", "charges", "forces", "energy")
    }
    for name, result in optional.items():
        status = result if isinstance(result, str) else result["status"]
        if status in {"lost", "changed", "introduced"}:
            warnings.append(f"{name} were {status} during conversion")
    if atom_count != "preserved":
        warnings.append("atom count changed")
    if species != "preserved":
        warnings.append("chemical symbols or their ordering changed")
    if positions["status"] != "preserved":
        warnings.append("atomic positions changed beyond tolerance")
    if cell["status"] != "preserved":
        warnings.append(f"cell status: {cell['status']}")
    if pbc != "preserved":
        warnings.append(f"periodic boundary-condition status: {pbc}")

    required_ok = all(
        (
            atom_count == "preserved",
            species == "preserved",
            positions["status"] == "preserved",
            cell["status"] == "preserved",
            pbc == "preserved",
        )
    )
    return {
        "source": workspace.relative(source_path),
        "destination": workspace.relative(destination_path),
        "source_format": source_format,
        "destination_format": destination_format,
        "tolerance": tolerance,
        "atom_count": atom_count,
        "species": species,
        "positions": positions,
        "cell": cell,
        "pbc": pbc,
        **optional,
        "required_structure_properties_preserved": required_ok,
        "warnings": warnings,
    }


def _array_comparison(
    before: np.ndarray, after: np.ndarray, tolerance: float, unit: str
) -> dict[str, Any]:
    if before.shape != after.shape:
        return {"status": "changed", "reason": "shape_changed", "unit": unit}
    max_error = float(np.max(np.abs(before - after))) if before.size else 0.0
    return {
        "status": "preserved" if max_error <= tolerance else "changed",
        f"max_error_{unit}": max_error,
    }


def _optional_comparison(before: Any, after: Any, tolerance: float) -> Any:
    if before is None and after is None:
        return "not_present"
    if before is not None and after is None:
        return "lost"
    if before is None and after is not None:
        return "introduced"
    if isinstance(before, list) or isinstance(after, list):
        return "preserved" if before == after else "changed"
    before_array = np.asarray(before)
    after_array = np.asarray(after)
    comparison = _array_comparison(before_array, after_array, tolerance, "absolute")
    return comparison


def validation_tool_specs(workspace: Workspace) -> list[ToolSpec]:
    return [
        ToolSpec(
            name="validate_conversion",
            description=(
                "Compare source and converted structures: atom count, ordered symbols, positions, "
                "cell, PBC, velocities, constraints, charges, forces, and energy."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "destination": {"type": "string"},
                    "tolerance": {"type": "number", "exclusiveMinimum": 0, "default": 1e-8},
                },
                "required": ["source", "destination"],
                "additionalProperties": False,
            },
            function=lambda source, destination, tolerance=1e-8: validate_conversion(
                workspace, source, destination, tolerance
            ),
            safety={"filesystem": "read", "workspace_only": True},
        )
    ]
