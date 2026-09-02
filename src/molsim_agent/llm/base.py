"""Interface between the agent runtime and language models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from molsim_agent.agent.messages import LLMResponse, Message


class LLMBackend(ABC):
    """A backend converts messages and tool schemas into one model response."""

    @abstractmethod
    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> LLMResponse:
        """Return either assistant text, tool calls, or both."""
