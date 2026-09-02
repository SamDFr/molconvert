"""Deterministic backend for learning and agent-loop tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from molsim_agent.agent.messages import LLMResponse, Message
from molsim_agent.llm.base import LLMBackend


class MockBackend(LLMBackend):
    """Return a predefined response script and retain every request for inspection."""

    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = iter(responses)
        self.requests: list[tuple[list[Message], list[dict[str, Any]]]] = []

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> LLMResponse:
        self.requests.append((list(messages), list(tools)))
        try:
            return next(self._responses)
        except StopIteration as exc:
            raise RuntimeError("MockBackend response script was exhausted") from exc
