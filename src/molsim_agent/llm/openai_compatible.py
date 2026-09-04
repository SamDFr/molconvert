"""Backend for OpenAI Chat Completions-compatible APIs.

This covers OpenAI, Mistral, OpenRouter, Together, Groq, and local servers such
as LM Studio when they expose ``/v1/chat/completions``.  The agent still owns
the loop; this class only translates one request and one response.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from molsim_agent.agent.messages import LLMResponse, Message, ToolCall
from molsim_agent.llm.base import LLMBackend


class APIError(RuntimeError):
    """Raised when a remote model API cannot provide a valid response."""


class OpenAICompatibleBackend(LLMBackend):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
    ) -> None:
        if not model.strip():
            raise ValueError("API model name cannot be empty")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Set an API key or OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, messages: Sequence[Message], tools: Sequence[dict[str, Any]]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [self._message(message) for message in messages],
            "tools": list(tools),
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise APIError(f"API returned HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, URLError) as exc:
            raise APIError(f"Could not connect to API at {self.base_url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise APIError("API returned invalid JSON") from exc
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise APIError("API response did not contain a chat message") from exc
        calls = []
        for index, raw in enumerate(message.get("tool_calls", [])):
            function = raw.get("function", {})
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append(ToolCall(str(raw.get("id") or f"api-call-{index}"), str(function.get("name", "")), arguments))
        return LLMResponse(content=str(message.get("content") or ""), tool_calls=calls)

    @staticmethod
    def _message(message: Message) -> dict[str, Any]:
        result: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.name:
            result["name"] = message.name
        if message.tool_call_id:
            result["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            result["tool_calls"] = [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}
                for call in message.tool_calls
            ]
        return result


class MistralBackend(OpenAICompatibleBackend):
    """Mistral API convenience backend."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        kwargs.setdefault("api_key", os.environ.get("MISTRAL_API_KEY"))
        kwargs.setdefault("base_url", "https://api.mistral.ai/v1")
        super().__init__(model, **kwargs)
