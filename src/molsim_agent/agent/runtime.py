"""Framework-free workflow dispatch for the explicit agent runtime."""
from __future__ import annotations

from collections.abc import Sequence

from molsim_agent.agent.workflows import ConversionWorkflow, ResearchWorkflow, Workflow


class AgentRuntime:
    """Select a domain workflow without embedding domain code in the loop."""

    def __init__(self, workflows: Sequence[Workflow] | None = None) -> None:
        self.workflows = tuple(workflows or (ConversionWorkflow(), ResearchWorkflow()))

    def workflow_for(self, objective: str) -> Workflow:
        return next(
            (workflow for workflow in self.workflows if workflow.matches(objective)),
            self.workflows[-1],
        )
