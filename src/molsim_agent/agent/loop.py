"""The deliberately explicit observe/tool/repeat agent loop."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from molsim_agent.agent.messages import Message, ToolCall
from molsim_agent.agent.intent import rewrite_objective
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
briefly, but do not reveal private chain-of-thought. If a requested format is not in the
implemented tool schema, say that it is not implemented here; never substitute a format."""

COMPACT_SYSTEM_PROMPT = """You are a molecular structure conversion agent.
Use the provided tools for evidence and all file operations. ASE—not you—handles atomic
data. Never invent scientific information. Validate every converted output before giving
a concise final report. Call the one available tool instead of describing a future plan.
Extended XYZ always means the tool argument target_format="extxyz", never "xyz". If a
requested format is absent from the tool schema, say that it is not implemented here;
never guess or substitute another format."""

PROFILES = ("full", "compact", "auto")
INTENT_MODES = ("deterministic", "llm")


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
        dry_run: bool = False,
        intent_mode: str = "deterministic",
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
        if intent_mode not in INTENT_MODES:
            raise ValueError(f"intent_mode must be one of: {', '.join(INTENT_MODES)}")
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
        self.dry_run = dry_run
        self.intent_mode = intent_mode
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
        effective_objective = objective
        if self.intent_mode == "llm":
            try:
                rewritten = rewrite_objective(self.backend, objective)
            except Exception:
                rewritten = None
            if rewritten:
                effective_objective = rewritten
                self._emit(
                    "intent_rewrite",
                    {"status": "rewritten", "original": objective, "rewritten": rewritten},
                )
            elif not rewritten:
                self._emit("intent_rewrite", {"status": "fallback"})
        state = AgentState(
            objective=effective_objective,
            original_objective=objective,
            dry_run=self.dry_run,
        )
        if self.intent_mode == "deterministic":
            intent = (
                self._compact_expected_arguments(state, "convert_structure")
                if self.profile == "compact" and self._is_conversion_objective(objective)
                else {"objective": objective}
            )
            self._emit("intent_rewrite", {"status": "deterministic", "intent": intent})
        state.messages.extend(
            [Message(role="system", content=self.system_prompt), Message(role="user", content=objective)]
        )
        empty_model_responses = 0
        blocked_model_responses = 0

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

            if not response.tool_calls and not response.content.strip():
                empty_model_responses += 1
                if empty_model_responses >= 2:
                    state.warnings.append(
                        "Stopped after repeated empty model responses; the provider did not "
                        "return a tool call or a final answer."
                    )
                    state.final_answer = (
                        "I could not continue because the model returned no tool call or "
                        "answer twice in a row. Try --no-think, a stronger tool-calling "
                        "model, or a larger timeout."
                    )
                    self._emit(
                        "limit",
                        {"iterations": iteration, "reason": "repeated_empty_response"},
                    )
                    return state
            elif response.tool_calls or response.content.strip():
                empty_model_responses = 0

            if (
                self.progress_level != "off"
                and response.tool_calls
                and not response.content.strip()
                and not any(
                    message.role == "assistant"
                    and not message.content.strip()
                    and message.tool_calls
                    and message.tool_calls[0].name == response.tool_calls[0].name
                    for message in state.messages[:-1]
                )
            ):
                # In progress mode, require the model to announce its next action
                # in its own words before executing a tool. This avoids fabricating
                # canned status text in the runtime.
                reminder = (
                    "Progress announcement required: before requesting a tool, write one "
                    "user-facing sentence saying what you found and what you will do next. "
                    "Use only known observations; do not reveal chain-of-thought."
                )
                state.messages.append(Message(role="system", content=reminder))
                self._emit("progress_required", {"tool": response.tool_calls[0].name})
                continue

            if not response.tool_calls:
                if not response.content.strip():
                    state.warnings.append("Model returned neither a tool call nor an answer")
                    continue
                blocker = self._completion_blocker(state)
                if blocker is not None:
                    blocked_model_responses += 1
                    if blocked_model_responses >= 2:
                        state.warnings.append(
                            "Stopped after repeated premature final answers; required "
                            "workflow tools were not called."
                        )
                        state.final_answer = (
                            "I could not continue because the model repeatedly returned a "
                            "final message before calling the required workflow tools. "
                            "Try --no-think or a model with stronger tool-calling support."
                        )
                        self._emit(
                            "limit",
                            {"iterations": iteration, "reason": "repeated_blocked_answer"},
                        )
                        return state
                    state.messages.append(Message(role="system", content=blocker))
                    self._emit("completion_blocked", {"reason": blocker})
                    continue
                blocked_model_responses = 0
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
                    workflow_error = self._workflow_error(state, call.name)
                    if workflow_error is not None:
                        raise ValueError(workflow_error)
                    self.registry.validate(call.name, call.arguments)
                    if self.dry_run and call.name == "convert_structure":
                        result = {
                            "dry_run": True,
                            "source": call.arguments.get("source"),
                            "destination": call.arguments.get("destination"),
                            "target_format": call.arguments.get("target_format"),
                            "message": "Write skipped because dry-run mode is enabled.",
                        }
                    else:
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
                if observation["ok"] and call.name in {
                    "detect_file_format", "inspect_structure", "convert_structure", "validate_conversion"
                }:
                    state.phase = {
                        "detect_file_format": "source_detected",
                        "inspect_structure": "structure_inspected",
                        "convert_structure": "converted" if not self.dry_run else "dry_run",
                        "validate_conversion": "validated",
                    }[call.name]
                if self.dry_run and call.name == "convert_structure" and observation["ok"]:
                    state.final_answer = (
                        "Dry run complete: no files were written. "
                        f"The planned output is {call.arguments.get('destination')}."
                    )
                    return state
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
        if self._is_conversion_objective(state.objective):
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

    def _workflow_error(self, state: AgentState, tool_name: str) -> str | None:
        """Keep conversion tools in a deterministic scientific order."""
        if not self._is_conversion_objective(state.objective):
            return None
        sequence = (
            "detect_file_format",
            "inspect_structure",
            "convert_structure",
            "validate_conversion",
        )
        successful = self._successful_tool_names(state)
        if tool_name == "convert_structure" and "inspect_structure" in successful:
            # Full-profile models may request several independent output formats
            # in one response (for example XYZ and CIF). Each is still validated
            # later by its own source/destination pair.
            return None
        next_tool = next((name for name in sequence if name not in successful), None)
        if next_tool is not None and tool_name != next_tool:
            return f"Workflow order requires {next_tool!r} before {tool_name!r}"
        return None

    def _tool_schemas_for_state(self, state: AgentState) -> list[dict[str, Any]]:
        schemas = deepcopy(self.registry.schemas())
        if self.profile != "compact" or not self._is_conversion_objective(state.objective):
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
            "phase": state.phase,
            "dry_run": state.dry_run,
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
            if next_tool is not None and self._is_conversion_objective(state.objective)
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
                "write exactly one concise scientific sentence (maximum 25 words) stating "
                "only observed facts and the next action; mention file, format, atom count, "
                "species, cell, or validation status only when available. No speculation, "
                "general explanations, or chain-of-thought."
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
            "extended XYZ means target_format='extxyz'; unsupported requested formats must "
            "be reported as not implemented.\n"
            f"Current action: {directive}"
        )
        intent = self._compact_expected_arguments(state, "convert_structure")
        if intent:
            compact_context += (
                "\nInterpreted conversion intent (runtime-normalized; use these values): "
                f"{json.dumps(intent, sort_keys=True)}"
            )
        if next_tool is not None:
            expected = self._compact_expected_arguments(state, next_tool)
            if expected:
                compact_context += (
                    "\nRequired argument values: "
                    f"{json.dumps(expected, sort_keys=True)}"
                )
        return [Message(role="user", content=compact_context)]

    @staticmethod
    def _is_conversion_objective(objective: str) -> bool:
        """Recognize common conversion verbs without making the LLM parse the task alone."""
        return bool(re.search(r"\bconvert\b", objective, re.IGNORECASE)) or bool(
            re.search(
                r"\b(?:convert|transform|export|write|save|turn|change)\b.*"
                r"\b(?:to|into|as)\b",
                objective,
                re.IGNORECASE | re.DOTALL,
            )
        )

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
        source_match = re.search(
            r"\b(?:convert|transform|export|write|save|turn|change)\s+(?:the\s+)?([^\s,]+)",
            objective,
            re.IGNORECASE,
        )
        if source_match is None:
            source_match = re.search(
                r"\bfrom\s+([^\s,]+)", objective, re.IGNORECASE
            )
        source = source_match.group(1).rstrip(".;") if source_match else None
        # English requests often say “find the POSCAR and convert it”. Resolve
        # the explicitly named standard input file from the workspace.
        if source in {None, "the", "it", "this", "that"} or not self._path_is_existing_file(source):
            mentioned = re.findall(
                r"\b(?:POSCAR|CONTCAR|[A-Za-z0-9_./-]+\.(?:xyz|extxyz|traj|data|cif))\b",
                objective,
                re.IGNORECASE,
            )
            mentioned = list(dict.fromkeys(
                name for name in mentioned if self._path_is_existing_file(name)
            ))
            candidates = mentioned or self._workspace_structure_candidates()
            if len(candidates) == 1:
                source = candidates[0]
        if source is not None:
            try:
                if not self.workspace.resolve(source).is_file():
                    source = None
            except ValueError:
                source = None
        destination_match = re.search(
            r"\b(?:as|to)\s+([A-Za-z0-9_./-]+\.(?:xyz|extxyz|traj|data|cif))",
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
            elif re.search(r"\blammps\b", lowered):
                expected["target_format"] = "lammps-data"
            elif destination and destination.lower().endswith(".data"):
                expected["target_format"] = "lammps-data"
            elif destination and destination.lower().endswith(".cif"):
                expected["target_format"] = "cif"
            elif re.search(r"\b(?:ase\s+traj|trajectory|traj)\b", lowered) or (
                destination and destination.lower().endswith(".traj")
            ):
                expected["target_format"] = "traj"
            elif re.search(r"\b(?:xyz|xyz file|xyz format)\b", lowered):
                expected["target_format"] = "xyz"
            elif re.search(r"\b(?:cif|crystallographic information file)\b", lowered):
                expected["target_format"] = "cif"
            if source and "target_format" in expected and not destination:
                # A filename is a safe, non-scientific default.  This keeps ordinary
                # requests such as "convert POSCAR to LAMMPS data" usable while
                # still leaving scientific mappings to the user.
                suffixes = {
                    "xyz": ".xyz",
                    "extxyz": ".extxyz",
                    "traj": ".traj",
                    "lammps-data": ".data",
                    "cif": ".cif",
                    "vasp": ".vasp",
                }
                source_path = Path(source)
                expected["destination"] = str(
                    source_path.with_name(source_path.name + suffixes[expected["target_format"]])
                )
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

    def _workspace_structure_candidates(self) -> list[str]:
        names: list[str] = []
        supported = {"POSCAR", "CONTCAR", ".xyz", ".extxyz", ".traj", ".data", ".cif"}
        for path in self.workspace.root.rglob("*"):
            if not path.is_file() or ".git" in path.parts:
                continue
            if path.name in supported or path.suffix.lower() in supported:
                names.append(self.workspace.relative(path))
        return names

    def _path_is_existing_file(self, name: str) -> bool:
        try:
            return self.workspace.resolve(name).is_file()
        except ValueError:
            return False
