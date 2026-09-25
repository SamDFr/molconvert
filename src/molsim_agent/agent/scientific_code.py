"""Opt-in ScientificCodeAgent boundary for temporary analysis implementations."""
from __future__ import annotations

import re
from pathlib import Path

from molsim_agent.agent.subagents import SubAgent, SubAgentResult
from molsim_agent.generated_tools.lifecycle import validate_generated_tool


class ScientificCodeAgent(SubAgent):
    """Create/validate temporary code without modifying trusted application code.

    Actual code synthesis is backend-specific and deliberately not automatic yet. This
    class gives the orchestrator a safe, testable hand-off point.
    """

    def __init__(self, root: str | Path, **kwargs) -> None:
        super().__init__(
            name="scientific_code_agent",
            description="Implements and tests missing scientific observables in isolation.",
            **kwargs,
        )
        self.root = Path(root).resolve()

    def task_directory(self, task_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("._") or "task"
        path = (self.root / safe).resolve()
        if self.root not in path.parents:
            raise ValueError("task directory escaped generated-tools root")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def validate(self, task_id: str, timeout: int = 30) -> SubAgentResult:
        result = validate_generated_tool(self.task_directory(task_id), timeout=timeout)
        if not result["ok"]:
            return SubAgentResult("rejected", "Generated tool failed validation.", warnings=[str(result)])
        return SubAgentResult("success", "Generated tool passed validation and is temporary.", created_tools=[task_id])
