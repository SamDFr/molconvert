"""Structured capability checks for scientific requests.

The orchestrator may understand a scientific objective even when no trusted tool can
execute it. Keeping that distinction explicit prevents hallucinated results.
"""
from __future__ import annotations

from molsim_agent.science.models import CapabilityAssessment, CapabilityStatus


def assess_capability(task: str, tool_names: set[str]) -> CapabilityAssessment:
    """Conservative baseline assessment used before scientific execution.

    Workflows can provide richer domain-specific assessors. This helper deliberately
    reports unknown analyses as implementable gaps instead of pretending they exist.
    """
    lowered = task.lower()
    if "vasp" in lowered and any(
        term in lowered for term in ("aimd", "incar", "potcar", "kpoints", "k-points", "input")
    ):
        return CapabilityAssessment(
            CapabilityStatus.NEEDS_USER_INPUT,
            task,
            reason=(
                "Reliable VASP/AIMD inputs depend on the pseudopotential set, electronic "
                "structure, spin state, k-point convergence, timestep, ensemble, and "
                "available VASP executable. These cannot be inferred safely from POSCAR alone."
            ),
            missing_capability="validated_vasp_workflow_builder",
            proposed_solution=(
                "Collect the missing scientific choices and pseudopotential metadata, then "
                "generate INCAR/KPOINTS/POTCAR with explicit user approval."
            ),
        )
    if "lammps" in lowered and any(term in lowered for term in ("md", "molecular dynamics", "input", "run")):
        return CapabilityAssessment(
            CapabilityStatus.NEEDS_USER_INPUT,
            task,
            reason="LAMMPS requires an explicit force field or ML potential; POSCAR does not contain one.",
            missing_capability="validated_lammps_workflow_builder",
            proposed_solution="Prepare a data file and input template with explicit pair_style/pair_coeff placeholders.",
        )
    aliases = {
        "single_point": ("single point", "energy and forces"),
        "geometry_optimization": ("geometry optimization", "optimize", "optimise"),
        "run_md": ("molecular dynamics", " md ", "nve", "nvt"),
        "trajectory_summary": ("trajectory summary", "summarize trajectory", "summarise trajectory"),
        "rdf": ("rdf", "radial distribution"),
        "msd": ("msd", "mean squared displacement"),
    }
    matched = next((name for name, words in aliases.items() if any(word in f" {lowered} " for word in words)), None)
    if matched and matched in tool_names:
        return CapabilityAssessment(CapabilityStatus.AVAILABLE, task)
    if any(word in lowered for word in ("correlation", "autocorrelation", "vacf", "residence time", "orientational")):
        return CapabilityAssessment(
            CapabilityStatus.NEEDS_IMPLEMENTATION,
            task,
            reason="No registered trusted tool computes this observable.",
            missing_capability=matched or "scientific_observable",
            proposed_solution="Implement and test the observable with NumPy/MDAnalysis.",
            delegate_to="scientific_code_agent",
        )
    return CapabilityAssessment(
        CapabilityStatus.UNSUPPORTED,
        task,
        reason="No registered capability matches this scientific request.",
    )
