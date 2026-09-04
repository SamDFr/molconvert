"""Explicit, inspectable state for a single agent task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from molsim_agent.agent.messages import Message, ToolCall


@dataclass(slots=True)
class ToolExecution:
    call: ToolCall
    result: dict[str, Any]


@dataclass(slots=True)
class AgentState:
    objective: str
    phase: str = "planning"
    dry_run: bool = False
    messages: list[Message] = field(default_factory=list)
    tool_executions: list[ToolExecution] = field(default_factory=list)
    created_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    iteration_count: int = 0
    final_answer: str | None = None
