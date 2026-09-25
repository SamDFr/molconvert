"""Small, isolated sub-agent abstraction (no orchestration framework)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from molsim_agent.llm.base import LLMBackend


@dataclass(slots=True)
class SubAgentResult:
    status: str
    summary: str
    artifacts: list[str] = field(default_factory=list)
    created_tools: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SubAgent:
    name: str
    description: str
    backend: LLMBackend | None = None
    system_prompt: str = ""
    allowed_tools: tuple[str, ...] = ()
    max_iterations: int = 8

    def run(self, task: str, context: dict[str, Any] | None = None) -> SubAgentResult:
        if self.backend is None:
            return SubAgentResult("needs_backend", f"{self.name} has no configured backend.")
        # Deliberately return a summary boundary; parent agents never receive the
        # sub-agent's private conversation transcript.
        return SubAgentResult(
            "not_implemented",
            f"{self.name} is available for delegation but has no workflow adapter yet.",
            warnings=["Sub-agent execution adapter is intentionally opt-in."],
        )
