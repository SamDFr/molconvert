"""LLM-based scientific task planning with strict runtime validation."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from molsim_agent.agent.messages import Message
from molsim_agent.llm.base import LLMBackend


@dataclass(frozen=True, slots=True)
class PlanStep:
    tool: str
    arguments: dict[str, Any]
    purpose: str = ""


@dataclass(frozen=True, slots=True)
class ScientificTaskPlan:
    objective: str
    steps: tuple[PlanStep, ...]
    missing_inputs: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()


def plan_task(
    backend: LLMBackend,
    objective: str,
    tool_schemas: list[dict[str, Any]],
    workspace_files: list[str],
) -> ScientificTaskPlan | None:
    """Ask the model for a tool plan; return None for an invalid/unavailable plan."""
    available = [schema.get("function", {}) for schema in tool_schemas]
    prompt = {
        "objective": objective,
        "workspace_files": workspace_files,
        "available_tools": available,
    }
    messages = [
        Message(
            role="system",
            content=(
                "You are the planning component of a scientific molecular-simulation agent. "
                "Understand the user's complete objective and decompose every requested "
                "subtask into a minimal ordered tool plan. Return JSON only with keys "
                "objective, steps, missing_inputs, assumptions. Each step must contain "
                "tool, arguments, and optional purpose. Use only tools listed in "
                "available_tools. Use explicit user values; otherwise omit arguments so "
                "trusted tools can apply documented defaults. Never invent coordinates, "
                "force fields, charges, units, files, or scientific results. Include an "
                "inspection step before modifying or simulating a structure when relevant. "
                "Do not substitute a molecular-dynamics template for geometry optimization, "
                "energy calculation, analysis, or another different scientific task. If no "
                "registered tool matches a requested subtask, return no step for that subtask "
                "and explain it in missing_inputs."
            ),
        ),
        Message(role="user", content=json.dumps(prompt, ensure_ascii=False)),
    ]
    response = backend.chat(messages, [])
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.content.strip(), flags=re.IGNORECASE)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        return None
    steps: list[PlanStep] = []
    for item in data["steps"]:
        if not isinstance(item, dict) or not isinstance(item.get("tool"), str):
            return None
        arguments = item.get("arguments", {})
        if not isinstance(arguments, dict):
            return None
        steps.append(PlanStep(item["tool"], arguments, str(item.get("purpose", ""))))
    missing = data.get("missing_inputs", [])
    assumptions = data.get("assumptions", [])
    if len(steps) > 20 or (not steps and not missing):
        return None
    return ScientificTaskPlan(
        objective=str(data.get("objective") or objective),
        steps=tuple(steps),
        missing_inputs=tuple(str(value) for value in missing) if isinstance(missing, list) else (),
        assumptions=tuple(str(value) for value in assumptions) if isinstance(assumptions, list) else (),
    )
