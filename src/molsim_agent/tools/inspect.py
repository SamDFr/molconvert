"""Model-facing structure detection and inspection tools."""

from __future__ import annotations

from molsim_agent.formats.detection import detect_format
from molsim_agent.formats.structures import read_structure, structure_summary
from molsim_agent.safety.policies import Workspace
from molsim_agent.tools.registry import ToolSpec


def detect_file_format(workspace: Workspace, path: str) -> dict[str, object]:
    source = workspace.resolve(path, must_exist=True)
    if not source.is_file():
        raise ValueError(f"Not a file: {path}")
    format_name, warnings = detect_format(source)
    return {
        "path": workspace.relative(source),
        "format": format_name,
        "supported": format_name is not None,
        "warnings": warnings,
    }


def inspect_structure(
    workspace: Workspace, path: str, *, include_coordinates: bool = True
) -> dict[str, object]:
    source = workspace.resolve(path, must_exist=True)
    atoms, format_name, warnings = read_structure(source)
    summary = structure_summary(atoms)
    if not include_coordinates:
        summary.pop("positions_angstrom")
        # Compact agents need counts and capabilities, not a repeated list of
        # every symbol; the deterministic tool still retains the full structure.
        summary.pop("chemical_symbols")
    return {
        "path": workspace.relative(source),
        "format": format_name,
        "structure": summary,
        "warnings": warnings,
    }


def inspection_tool_specs(
    workspace: Workspace, *, include_coordinates: bool = True
) -> list[ToolSpec]:
    path_parameter = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Workspace-relative file path."}},
        "required": ["path"],
        "additionalProperties": False,
    }
    return [
        ToolSpec(
            "detect_file_format",
            "Detect whether a file is a supported VASP, XYZ/extXYZ, ASE traj, or LAMMPS data file.",
            path_parameter,
            lambda path: detect_file_format(workspace, path),
            {"filesystem": "read", "workspace_only": True},
        ),
        ToolSpec(
            "inspect_structure",
            "Parse one structure with ASE and report atoms, species, coordinates, cell, PBC, and optional properties.",
            path_parameter,
            lambda path: inspect_structure(
                workspace, path, include_coordinates=include_coordinates
            ),
            {"filesystem": "read", "workspace_only": True},
        ),
    ]
