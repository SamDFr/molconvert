"""Minimal Anthropic Messages API backend (Claude)."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from molsim_agent.agent.messages import LLMResponse, Message, ToolCall
from molsim_agent.llm.base import LLMBackend
from molsim_agent.llm.openai_compatible import APIError


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str, *, api_key: str | None = None, base_url: str = "https://api.anthropic.com", timeout: float = 120.0) -> None:
        if not model.strip():
            raise ValueError("Anthropic model name cannot be empty")
        self.model, self.api_key, self.base_url, self.timeout = model, api_key or os.environ.get("ANTHROPIC_API_KEY"), base_url.rstrip("/"), timeout
        if not self.api_key:
            raise ValueError("Set an API key or ANTHROPIC_API_KEY")

    def chat(self, messages: Sequence[Message], tools: Sequence[dict[str, Any]]) -> LLMResponse:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        payload: dict[str, Any] = {"model": self.model, "max_tokens": 4096, "messages": [self._message(m) for m in messages if m.role != "system"], "tools": [self._tool(t) for t in tools]}
        if system:
            payload["system"] = system
        request = Request(f"{self.base_url}/v1/messages", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
        except HTTPError as exc:
            raise APIError(f"Anthropic API returned HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
        except (TimeoutError, URLError) as exc:
            raise APIError(f"Could not connect to Anthropic API: {exc}") from exc
        blocks = body.get("content", [])
        text = "\n".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text")
        calls = [ToolCall(str(b.get("id", f"claude-call-{i}")), str(b.get("name", "")), b.get("input", {})) for i, b in enumerate(blocks) if b.get("type") == "tool_use"]
        return LLMResponse(content=text, tool_calls=calls)

    @staticmethod
    def _tool(schema: dict[str, Any]) -> dict[str, Any]:
        function = schema["function"]
        return {"name": function["name"], "description": function.get("description", ""), "input_schema": function.get("parameters", {"type": "object"})}

    @staticmethod
    def _message(message: Message) -> dict[str, Any]:
        if message.role == "tool":
            return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": message.tool_call_id, "content": message.content}]}
        if message.role == "assistant" and message.tool_calls:
            content: list[dict[str, Any]] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            content.extend({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in message.tool_calls)
            return {"role": "assistant", "content": content}
        return {"role": message.role, "content": message.content}
