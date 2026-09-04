"""Small explicit registry mapping model-visible names to Python callables."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


ToolFunction = Callable[..., dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    function: ToolFunction = field(repr=False)
    safety: dict[str, Any] = field(default_factory=dict)

    def model_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.model_schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            tool = self._tools[name]
        except KeyError as exc:
            raise ValueError(f"Unknown tool: {name}") from exc
        self._validate_arguments(tool, arguments)
        result = tool.function(**arguments)
        if not isinstance(result, dict):
            raise TypeError(f"Tool {name} must return a dictionary")
        return result

    def validate(self, name: str, arguments: dict[str, Any]) -> None:
        """Validate a model call without executing its Python function."""
        try:
            tool = self._tools[name]
        except KeyError as exc:
            raise ValueError(f"Unknown tool: {name}") from exc
        self._validate_arguments(tool, arguments)

    @staticmethod
    def _validate_arguments(tool: ToolSpec, arguments: dict[str, Any]) -> None:
        if not isinstance(arguments, dict):
            raise TypeError("Tool arguments must be an object")
        properties = tool.parameters.get("properties", {})
        required = tool.parameters.get("required", [])
        missing = [key for key in required if key not in arguments]
        if missing:
            raise ValueError(f"Missing required arguments: {', '.join(missing)}")
        unknown = set(arguments) - set(properties)
        if unknown and tool.parameters.get("additionalProperties") is False:
            raise ValueError(f"Unknown arguments: {', '.join(sorted(unknown))}")
        json_types = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        for key, value in arguments.items():
            definition = properties.get(key, {})
            expected_name = definition.get("type")
            expected = json_types.get(expected_name)
            if expected is not None and (
                not isinstance(value, expected)
                or expected_name in {"integer", "number"} and isinstance(value, bool)
            ):
                raise TypeError(f"Argument {key!r} must have JSON type {expected_name}")
            if "enum" in definition and value not in definition["enum"]:
                raise ValueError(f"Argument {key!r} must be one of {definition['enum']}")
