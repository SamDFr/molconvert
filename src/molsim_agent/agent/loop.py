"""The deliberately explicit observe/tool/repeat agent loop."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from molsim_agent.agent.messages import Message, ToolCall
from molsim_agent.agent.planner import build_system_prompt
from molsim_agent.agent.state import AgentState, ToolExecution
from molsim_agent.llm.base import LLMBackend
from molsim_agent.safety.policies import Workspace
from molsim_agent.tools.filesystem import filesystem_tool_specs
from molsim_agent.tools.convert import conversion_tool_specs
from molsim_agent.tools.inspect import inspection_tool_specs
from molsim_agent.tools.registry import ToolRegistry
from molsim_agent.tools.validate import validation_tool_specs


DEFAULT_SYSTEM_PROMPT = """You are a molecular-simulation conversion agent.
Use deterministic tools for all file inspection and conversion. Never invent scientific
data. Call tools when evidence is needed. Report preservation, loss, warnings, and
assumptions. Finish with a concise answer only after required validation is complete.
All paths are relative to the constrained workspace. Explain decisions and evidence
briefly, but do not reveal private chain-of-thought."""

COMPACT_SYSTEM_PROMPT = """You are a molecular structure conversion agent.
Use the provided tools for evidence and all file operations. ASE—not you—handles atomic
data. Never invent scientific information. Validate every converted output before giving
a concise final report. Call the one available tool instead of describing a future plan.
Extended XYZ always means the tool argument target_format="extxyz", never "xyz"."""

PROFILES = ("full", "compact", "auto")


class Agent:
    """Coordinates an LLM and deterministic tools without an agent framework."""

    def __init__(
        self,
        *,
        backend: LLMBackend | None = None,
        workspace: str | Path = ".",
        registry: ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 12,
        event_handler: Callable[[str, dict[str, Any]], None] | None = None,
        model: str | None = None,
        skill_paths: Sequence[str | Path] | None = None,
        profile: str = "full",
        progress: bool = False,
        progress_level: str | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if backend is None:
            if model is None:
                raise ValueError("Provide either an LLM backend or an Ollama model name")
            from molsim_agent.llm.ollama import OllamaBackend

            backend = OllamaBackend(model=model)
        self.backend = backend
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of: {', '.join(PROFILES)}")
        self.profile = self._resolve_profile(profile, backend)
        self.workspace = Workspace.from_path(workspace)
        self.registry = registry or self._default_registry()
        base_prompt = system_prompt or (
            COMPACT_SYSTEM_PROMPT if self.profile == "compact" else DEFAULT_SYSTEM_PROMPT
        )
        self.system_prompt = build_system_prompt(
            base_prompt, skill_paths, profile=self.profile
        )
        self.max_iterations = max_iterations
        self.event_handler = event_handler
        self.model = model
        self.progress_level = progress_level or ("brief" if progress else "off")
        if self.progress_level not in {"off", "brief", "detailed"}:
            raise ValueError("progress_level must be off, brief, or detailed")

    def _default_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        if self.profile == "compact":
            tool_groups = (
                inspection_tool_specs(self.workspace, include_coordinates=False),
                conversion_tool_specs(self.workspace),
                validation_tool_specs(self.workspace),
            )
        else:
            tool_groups = (
                filesystem_tool_specs(self.workspace),
                inspection_tool_specs(self.workspace),
                conversion_tool_specs(self.workspace),
                validation_tool_specs(self.workspace),
            )
        for group in tool_groups:
            for tool in group:
                registry.register(tool)
        return registry

    def run(self, objective: str) -> AgentState:
        state = AgentState(objective=objective)
        state.messages.extend(
            [Message(role="system", content=self.system_prompt), Message(role="user", content=objective)]
        )

        for iteration in range(1, self.max_iterations + 1):
            state.iteration_count = iteration
            self._emit("iteration", {"number": iteration})
            model_messages = self._messages_for_model(state)
            tool_schemas = self._tool_schemas_for_state(state)
            available_tools = {
                schema["function"]["name"] for schema in tool_schemas
            }
            registered_tools = {
                schema["function"]["name"] for schema in self.registry.schemas()
            }
            self._emit(
                "model_request",
                {
                    "messages": [
                        {"role": message.role, "content": message.content}
                        for message in model_messages
                    ],
                    "tools": tool_schemas,
                },
            )
            response = self.backend.chat(model_messages, tool_schemas)
            self._emit(
                "model_response",
                {
                    "content": response.content,
                    "tool_calls": [
                        {"name": call.name, "arguments": call.arguments}
                        for call in response.tool_calls
                    ],
                },
            )
            assistant = Message(
                role="assistant", content=response.content, tool_calls=response.tool_calls
            )
            state.messages.append(assistant)

            if not response.tool_calls:
                if not response.content.strip():
                    state.warnings.append("Model returned neither a tool call nor an answer")
                    continue
                blocker = self._completion_blocker(state)
                if blocker is not None:
                    state.messages.append(Message(role="system", content=blocker))
                    self._emit("completion_blocked", {"reason": blocker})
                    continue
                state.final_answer = response.content
                self._emit("final", {"content": response.content})
                return state

            for call in response.tool_calls:
                self._emit("tool_call", {"name": call.name, "arguments": call.arguments})
                try:
                    if call.name in registered_tools and call.name not in available_tools:
                        raise ValueError(
                            f"Tool {call.name!r} is not available in this agent step"
                        )
                    argument_error = self._compact_argument_error(state, call)
                    if argument_error is not None:
                        raise ValueError(argument_error)
                    result = self.registry.execute(call.name, call.arguments)
                    observation = {"ok": True, "result": result}
                except Exception as exc:  # tool failures are observations, not runtime crashes
                    observation = {
                        "ok": False,
                        "error": type(exc).__name__,
                        "message": str(exc),
                    }
                state.tool_executions.append(ToolExecution(call=call, result=observation))
                if observation["ok"]:
                    tool_result = observation["result"]
                    state.created_files.extend(tool_result.get("created_files", []))
                    state.modified_files.extend(tool_result.get("modified_files", []))
                    state.warnings.extend(tool_result.get("warnings", []))
                state.messages.append(
                    Message(
                        role="tool",
                        name=call.name,
                        tool_call_id=call.id,
                        content=json.dumps(observation, sort_keys=True),
                    )
                )
                self._emit("tool_result", {"name": call.name, "observation": observation})
                if not observation["ok"] and any(
                    previous.call.name == call.name
                    and previous.call.arguments == call.arguments
                    and not previous.result.get("ok")
                    for previous in state.tool_executions[:-1]
                ):
                    state.warnings.append("Stopped after a repeated failed tool call")
                    state.final_answer = (
                        f"I could not continue because tool {call.name!r} repeated the "
                        "same failed request. Check the file path and workspace."
                    )
                    self._emit("limit", {"iterations": iteration, "reason": "repeated_failure"})
                    return state

        state.warnings.append(f"Stopped after maximum of {self.max_iterations} iterations")
        state.final_answer = (
            f"I could not complete the task within {self.max_iterations} agent iterations."
        )
        self._emit("limit", {"iterations": self.max_iterations})
        return state

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self.event_handler is not None:
            self.event_handler(event, payload)

    @staticmethod
    def _resolve_profile(profile: str, backend: LLMBackend) -> str:
        if profile != "auto":
            return profile
        return "compact" if backend.__class__.__name__ == "OllamaBackend" else "full"

    def _completion_blocker(self, state: AgentState) -> str | None:
        successful_tools = self._successful_tool_names(state)
        if self.profile == "compact" and "convert" in state.objective.lower():
            required_order = (
                "detect_file_format",
                "inspect_structure",
                "convert_structure",
                "validate_conversion",
            )
            for required in required_order:
                if required not in successful_tools:
                    return (
                        "Compact conversion policy: do not answer yet. Call the currently "
                        f"available tool {required!r}."
                    )
        converted: set[tuple[str, str]] = set()
        validated: set[tuple[str, str]] = set()
        for execution in state.tool_executions:
            observation = execution.result
            if not observation.get("ok"):
                continue
            result = observation.get("result", {})
            if execution.call.name == "convert_structure":
                converted.add((str(result.get("source")), str(result.get("destination"))))
            elif execution.call.name == "validate_conversion":
                validated.add((str(result.get("source")), str(result.get("destination"))))
        pending = converted - validated
        if not pending:
            return None
        pairs = ", ".join(f"{source} -> {destination}" for source, destination in sorted(pending))
        return (
            "Runtime policy: you cannot finish yet. Call validate_conversion for: "
            f"{pairs}. Then report its actual result."
        )

    def _tool_schemas_for_state(self, state: AgentState) -> list[dict[str, Any]]:
        schemas = deepcopy(self.registry.schemas())
        if self.profile != "compact" or "convert" not in state.objective.lower():
            return schemas
        successful_tools = self._successful_tool_names(state)
        sequence = (
            "detect_file_format",
            "inspect_structure",
            "convert_structure",
            "validate_conversion",
        )
        next_tool = next(
            (name for name in sequence if name not in successful_tools), None
        )
        if next_tool is None:
            return []
        selected = [
            schema for schema in schemas if schema["function"]["name"] == next_tool
        ]
        expected = self._compact_expected_arguments(state, next_tool)
        for schema in selected:
            properties = schema["function"]["parameters"].get("properties", {})
            for name, value in expected.items():
                if name in properties:
                    properties[name]["const"] = value
        return selected

    @staticmethod
    def _successful_tool_names(state: AgentState) -> set[str]:
        return {
            execution.call.name
            for execution in state.tool_executions
            if execution.result.get("ok")
        }

    def _messages_for_model(self, state: AgentState) -> list[Message]:
        runtime_state = {
            "objective": state.objective,
            "iteration": state.iteration_count,
            "created_files": state.created_files,
            "modified_files": state.modified_files,
            "warnings": state.warnings,
        }
        state_message = Message(
            role="system",
            content=f"Current runtime state:\n{json.dumps(runtime_state, sort_keys=True)}",
        )
        if self.profile != "compact":
            return [state.messages[0], state_message, *state.messages[1:]]

        successful = self._successful_tool_names(state)
        sequence = (
            "detect_file_format",
            "inspect_structure",
            "convert_structure",
            "validate_conversion",
        )
        next_tool = next((name for name in sequence if name not in successful), None)
        directive = (
            f"Call {next_tool} now using the native tool interface. Do not write a plan or "
            "describe the call in text."
            if next_tool is not None and "convert" in state.objective.lower()
            else "All required tools completed. Give the concise scientific report now."
        )
        if self.progress_level == "brief" and next_tool is not None:
            directive = (
                f"Call {next_tool} now using the native tool interface. Before the tool call, "
                "write exactly one brief user-facing progress sentence; do not reveal "
                "chain-of-thought or a multi-step plan."
            )
        elif self.progress_level == "detailed" and next_tool is not None:
            directive = (
                f"Call {next_tool} now using the native tool interface. Before the tool call, "
                "write one or two factual user-facing sentences describing what has been "
                "observed so far (file name, format, atom count, species, cell, or validation "
                "status when available). Use only the tool observations above; never guess, "
                "and do not reveal chain-of-thought or a future multi-step plan."
            )
        observations = [
            {
                "tool": execution.call.name,
                "observation": execution.result,
            }
            for execution in state.tool_executions
        ]
        if next_tool is None:
            observations = [
                observation
                for observation in observations
                if observation["tool"] == "validate_conversion"
            ]
        compact_context = (
            f"Objective: {state.objective}\n"
            f"Tool observations: {json.dumps(observations, sort_keys=True)}\n"
            "Rules: tools and ASE handle atomic data; never invent data; never overwrite; "
            "extended XYZ means target_format='extxyz'.\n"
            f"Current action: {directive}"
        )
        if next_tool is not None:
            expected = self._compact_expected_arguments(state, next_tool)
            if expected:
                compact_context += (
                    "\nRequired argument values: "
                    f"{json.dumps(expected, sort_keys=True)}"
                )
        return [Message(role="user", content=compact_context)]

    def _compact_argument_error(
        self, state: AgentState, call: ToolCall
    ) -> str | None:
        if self.profile != "compact":
            return None
        expected = self._compact_expected_arguments(state, call.name)
        mismatches = {
            name: value
            for name, value in expected.items()
            if call.arguments.get(name) != value
        }
        if not mismatches:
            return None
        return (
            "Arguments do not match the explicit user objective. Required values: "
            f"{json.dumps(expected, sort_keys=True)}"
        )

    def _compact_expected_arguments(
        self, state: AgentState, tool_name: str
    ) -> dict[str, Any]:
        objective = state.objective
        source_match = re.search(r"\bconvert\s+([^\s,]+)", objective, re.IGNORECASE)
        source = source_match.group(1).rstrip(".;") if source_match else None
        # English requests often say “find the POSCAR and convert it”. Resolve
        # the explicitly named standard input file from the workspace.
        if source in {None, "it", "this", "that"}:
            for name in ("POSCAR", "CONTCAR"):
                if re.search(rf"\b{name}\b", objective, re.IGNORECASE):
                    try:
                        if self.workspace.resolve(name).is_file():
                            source = name
                            break
                    except ValueError:
                        pass
        if source is not None:
            try:
                if not self.workspace.resolve(source).is_file():
                    source = None
            except ValueError:
                source = None
        destination_match = re.search(
            r"\b(?:as|to)\s+([A-Za-z0-9_./-]+\.(?:xyz|extxyz|traj|data))",
            objective,
            re.IGNORECASE,
        )
        destination = destination_match.group(1) if destination_match else None

        if tool_name in {"detect_file_format", "inspect_structure"} and source:
            return {"path": source}
        if tool_name == "convert_structure":
            expected: dict[str, Any] = {}
            if source:
                expected["source"] = source
            if destination:
                expected["destination"] = destination
            lowered = objective.lower()
            if "extended xyz" in lowered or "extxyz" in lowered:
                expected["target_format"] = "extxyz"
            elif destination and destination.lower().endswith(".xyz"):
                expected["target_format"] = "xyz"
            elif "lammps" in lowered:
                expected["target_format"] = "lammps-data"
            elif "ase traj" in lowered or ".traj" in lowered:
                expected["target_format"] = "traj"
            return expected
        if tool_name == "validate_conversion":
            for execution in reversed(state.tool_executions):
                if execution.call.name == "convert_structure" and execution.result.get("ok"):
                    result = execution.result["result"]
                    return {
                        "source": result["source"],
                        "destination": result["destination"],
                    }
        return {}
