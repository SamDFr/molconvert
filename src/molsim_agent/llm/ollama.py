"""Ollama backend using its local HTTP chat API."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from molsim_agent.agent.messages import LLMResponse, Message, ToolCall
from molsim_agent.llm.base import LLMBackend


class OllamaError(RuntimeError):
    """Raised when the local Ollama service cannot provide a valid response."""


class OllamaBackend(LLMBackend):
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
        think: bool | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("Ollama model name cannot be empty")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.think = think

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [self._message_to_ollama(message) for message in messages],
            "tools": list(tools),
            "stream": False,
        }
        if self.think is not None:
            payload["think"] = self.think
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except TimeoutError as exc:
            raise OllamaError(
                f"Ollama did not respond within {self.timeout:g} seconds"
            ) from exc
        except URLError as exc:
            raise OllamaError(
                f"Could not connect to Ollama at {self.base_url}. Is `ollama serve` running?"
            ) from exc
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama returned invalid JSON") from exc

        try:
            message = body["message"]
        except (KeyError, TypeError) as exc:
            raise OllamaError("Ollama response did not contain a message") from exc

        calls = []
        for index, raw_call in enumerate(message.get("tool_calls", [])):
            function = raw_call.get("function", {})
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise OllamaError("Ollama returned malformed tool arguments") from exc
            calls.append(
                ToolCall(
                    id=str(raw_call.get("id") or f"ollama-call-{index}"),
                    name=str(function.get("name", "")),
                    arguments=arguments,
                )
            )
        content = str(message.get("content", ""))
        # Some small instruction-tuned models (notably qwen2.5-coder:1.5b/3b)
        # emit a tool request as JSON text instead of Ollama's native
        # ``message.tool_calls`` field.  Accept only the documented shape and
        # only names present in the schemas; ordinary JSON answers remain text.
        if not calls:
            fallback = self._parse_text_tool_call(content, tools)
            if fallback is not None:
                calls = [fallback]
                content = ""
        return LLMResponse(content=content, tool_calls=calls)

    @staticmethod
    def _parse_text_tool_call(
        content: str, tools: Sequence[dict[str, Any]]
    ) -> ToolCall | None:
        candidate = content.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if len(lines) >= 3:
                candidate = "\n".join(lines[1:-1]).strip()
        try:
            payload = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            return None
        schemas = {
            schema.get("function", {}).get("name"): schema
            for schema in tools
            if isinstance(schema, dict)
        }
        name = payload["name"]
        schema = schemas.get(name)
        if schema is None:
            return None
        raw_arguments = payload.get("arguments", {})
        if isinstance(raw_arguments, list):
            properties = schema.get("function", {}).get("parameters", {}).get("properties", {})
            keys = list(properties)
            if len(raw_arguments) > len(keys):
                return None
            arguments = dict(zip(keys, raw_arguments))
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            return None
        return ToolCall(id="ollama-text-call-0", name=name, arguments=arguments)

    @staticmethod
    def _message_to_ollama(message: Message) -> dict[str, Any]:
        result: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": call.id,
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in message.tool_calls
            ]
        if message.role == "tool" and message.name:
            result["tool_name"] = message.name
        return result
