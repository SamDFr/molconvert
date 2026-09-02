"""Backend-neutral messages exchanged by the agent and an LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ToolCall:
    """A model request to invoke one registered Python tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class Message:
    """One conversation item stored in agent state."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(slots=True)
class LLMResponse:
    """The normalized result of one backend chat request."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
