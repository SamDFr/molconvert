"""Generic registry for scientific input-template workflows.

The orchestrator knows only how to match, assess, execute, and report a template. Code-
specific defaults live in registered handlers, not in the central agent loop.
"""
from __future__ import annotations

from dataclasses import dataclass
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

    def execute(self, workflow: TemplateWorkflow, registry: ToolRegistry, source: str = "POSCAR") -> dict[str, Any]:
        return registry.execute(workflow.tool_name, {"source": source})


DEFAULT_TEMPLATE_WORKFLOWS = TemplateWorkflowRegistry(
    (
        TemplateWorkflow("vasp_aimd", ("vasp", "aimd"), "prepare_vasp_aimd_inputs", "VASP AIMD template"),
        TemplateWorkflow("lammps_md", ("lammps", "md"), "prepare_lammps_md_inputs", "LAMMPS MD template"),
    )
)
