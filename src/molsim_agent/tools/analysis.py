"""Workspace-safe deterministic trajectory analysis tools."""
from __future__ import annotations

from molsim_agent.analysis.basic import mean_squared_displacement, pair_distance_statistics, trajectory_summary
from molsim_agent.safety.policies import Workspace
from molsim_agent.tools.registry import ToolSpec


def analysis_tool_specs(workspace: Workspace) -> list[ToolSpec]:
    path = {"type": "string", "description": "Workspace-relative trajectory path."}
    return [
        ToolSpec(
            "trajectory_summary",
            "Summarize frames, atom count, and species using ASE.",
            {"type": "object", "properties": {"path": path}, "required": ["path"], "additionalProperties": False},
            lambda path: trajectory_summary(workspace.resolve(path, must_exist=True)),
            {"filesystem": "read", "workspace_only": True},
            category="analysis", risk="read", requirements=("ase",), compute_cost="cpu", deterministic=True,
        ),
        ToolSpec(
            "pair_distance_statistics",
            "Compute deterministic pair-distance summary for a trajectory.",
            {"type": "object", "properties": {"path": path, "cutoff_angstrom": {"type": "number"}}, "required": ["path"], "additionalProperties": False},
            lambda path, cutoff_angstrom=10.0: pair_distance_statistics(workspace.resolve(path, must_exist=True), cutoff_angstrom),
            {"filesystem": "read", "workspace_only": True},
            category="analysis", risk="read", requirements=("ase", "numpy"), compute_cost="cpu", deterministic=True,
        ),
        ToolSpec(
            "mean_squared_displacement",
            "Compute MSD relative to the first trajectory frame.",
            {"type": "object", "properties": {"path": path}, "required": ["path"], "additionalProperties": False},
            lambda path: mean_squared_displacement(workspace.resolve(path, must_exist=True)),
            {"filesystem": "read", "workspace_only": True},
            category="analysis", risk="read", requirements=("ase", "numpy"), compute_cost="cpu", deterministic=True,
        ),
    ]
