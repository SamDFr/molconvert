"""Generic registry for scientific input-template workflows.

The orchestrator knows only how to match, assess, execute, and report a template. Code-
specific defaults live in registered handlers, not in the central agent loop.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable

from molsim_agent.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class TemplateWorkflow:
    name: str
    keywords: tuple[str, ...]
    tool_name: str
    description: str

    def matches(self, objective: str) -> bool:
        lowered = objective.lower()
        return all(keyword in lowered for keyword in self.keywords)


class TemplateWorkflowRegistry:
    def __init__(self, workflows: tuple[TemplateWorkflow, ...] = ()) -> None:
        self.workflows = workflows

    def match(self, objective: str) -> TemplateWorkflow | None:
        return next((workflow for workflow in self.workflows if workflow.matches(objective)), None)

    def execute(self, workflow: TemplateWorkflow, registry: ToolRegistry, objective: str) -> dict[str, Any]:
        return registry.execute(workflow.tool_name, template_arguments(objective))


def template_arguments(objective: str) -> dict[str, Any]:
    """Extract only explicit protocol quantities; tools supply safe defaults otherwise."""
    lowered = objective.lower()
    arguments: dict[str, Any] = {"source": "POSCAR"}
    temperature = re.search(r"(\d+(?:\.\d+)?)\s*k\b", lowered)
    duration = re.search(r"(\d+(?:\.\d+)?)\s*ps\b", lowered)
    timestep = re.search(r"(?:timestep|step|pas\s+de\s+temps)[^\d]{0,20}(\d+(?:\.\d+)?)\s*fs\b", lowered)
    if temperature:
        arguments["temperature_K"] = float(temperature.group(1))
    if duration:
        arguments["duration_ps"] = float(duration.group(1))
    if timestep:
        arguments["timestep_fs"] = float(timestep.group(1))
    if re.search(r"\bnve\b", lowered):
        arguments["ensemble"] = "nve"
    elif re.search(r"\bnvt\b", lowered):
        arguments["ensemble"] = "nvt"
    return arguments


DEFAULT_TEMPLATE_WORKFLOWS = TemplateWorkflowRegistry(
    (
        TemplateWorkflow("vasp_aimd", ("vasp", "aimd"), "prepare_vasp_aimd_inputs", "VASP AIMD template"),
        TemplateWorkflow("lammps_md", ("lammps", "md"), "prepare_lammps_md_inputs", "LAMMPS MD template"),
        TemplateWorkflow("gromacs_md", ("gromacs", "md"), "prepare_gromacs_md_inputs", "GROMACS MD template"),
    )
)
