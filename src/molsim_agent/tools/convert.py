"""Deterministic, policy-constrained ASE structure conversion."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms

from molsim_agent.formats.detection import normalize_format
from molsim_agent.formats.structures import read_structure, write_structure
from molsim_agent.safety.policies import Workspace
from molsim_agent.tools.registry import ToolSpec


FORMAT_LIMITATIONS: dict[str, list[str]] = {
    "xyz": ["cell", "pbc", "velocities", "constraints", "charges", "forces", "energy"],
    "vasp": ["pbc", "constraints", "forces", "energy", "charges"],
    "lammps-data": ["pbc", "velocities", "constraints", "charges", "forces", "energy"],
    "extxyz": ["constraints"],
    "traj": [],
    "cif": ["velocities", "constraints", "charges", "forces", "energy"],
}


def convert_structure(
    workspace: Workspace,
    source: str,
    destination: str,
    target_format: str,
    overwrite: bool = False,
) -> dict[str, object]:
    source_path = workspace.resolve(source, must_exist=True)
    destination_path = workspace.resolve(destination)
    if source_path == destination_path:
        raise ValueError("Source and destination must be different files")
    requested_destination = workspace.relative(destination_path)
    destination_adjusted = False
    if destination_path.exists() and not overwrite:
        destination_path = _next_available_destination(destination_path)
        destination_adjusted = True
    existed_before = destination_path.exists()
    directory_created = False
    if not destination_path.parent.is_dir():
        # Creating a missing destination directory is a safe, reversible workspace
        # operation. The workspace boundary was already checked by resolve().
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        directory_created = True

    atoms, source_format, read_warnings = read_structure(source_path)
    normalized_target = normalize_format(target_format)
    write_structure(destination_path, atoms, normalized_target)
    warnings = list(read_warnings)
    if destination_adjusted:
        warnings.append(
            f"Requested destination {requested_destination} already existed; wrote a new file "
            f"at {workspace.relative(destination_path)} instead."
        )
    if directory_created:
        warnings.append(
            f"Created destination directory {workspace.relative(destination_path.parent)}."
        )
    limitations = FORMAT_LIMITATIONS[normalized_target]
    present = _present_properties(atoms)
    at_risk = [name for name in limitations if present[name]]
    if at_risk:
        warnings.append(
            f"The {normalized_target} format may not preserve detected properties: "
            f"{', '.join(at_risk)}; run validation."
        )
    return {
        "source": workspace.relative(source_path),
        "destination": workspace.relative(destination_path),
        "requested_destination": requested_destination,
        "destination_adjusted": destination_adjusted,
        "source_format": source_format,
        "target_format": normalized_target,
        "atom_count": len(atoms),
        "conversion_kind": "structure_format_conversion",
        "conversion_fidelity": (
            "potentially_lossy" if at_risk else "exact_for_detected_properties_pending_validation"
        ),
        "format_capability_limitations": limitations,
        "detected_properties_at_risk": at_risk,
        "warnings": warnings,
        "created_files": [] if existed_before else [workspace.relative(destination_path)],
        "modified_files": [workspace.relative(destination_path)] if existed_before else [],
    }


def _next_available_destination(path: Path) -> Path:
    """Return ``path`` or a numbered sibling without overwriting an existing file."""
    if not path.exists():
        return path
    suffix = path.suffix
    stem = path.name[: -len(suffix)] if suffix else path.name
    index = 1
    while True:
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _present_properties(atoms: Atoms) -> dict[str, bool]:
    from molsim_agent.formats.structures import optional_properties

    properties = optional_properties(atoms)
    return {
        "cell": bool(atoms.cell.any()),
        "pbc": bool(atoms.pbc.any()),
        **{name: value is not None for name, value in properties.items()},
    }


def conversion_tool_specs(workspace: Workspace) -> list[ToolSpec]:
    return [
        ToolSpec(
            name="convert_structure",
            description=(
                "Convert an existing atomic structure deterministically with ASE. Never use this "
                "for semantic workflow translation. Existing destinations are protected by default."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "destination": {"type": "string"},
                    "target_format": {
                        "type": "string",
                        "enum": ["vasp", "xyz", "extxyz", "traj", "lammps-data", "cif"],
                    },
                    "overwrite": {
                        "type": "boolean",
                        "default": False,
                        "description": "True only when the user explicitly authorized overwriting.",
                    },
                },
                "required": ["source", "destination", "target_format"],
                "additionalProperties": False,
            },
            function=lambda source, destination, target_format, overwrite=False: convert_structure(
                workspace, source, destination, target_format, overwrite
            ),
            safety={
                "filesystem": "write",
                "workspace_only": True,
                "overwrite_requires_explicit_argument": True,
            },
            category="conversion", risk="write", requirements=("ase",), compute_cost="cpu", deterministic=True,
        )
    ]
