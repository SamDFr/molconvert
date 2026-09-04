"""Optional LLM-assisted intent normalization.

The normalizer only prepares tool arguments; it never executes tools or invents
scientific mappings. The deterministic parser remains the default and fallback.
"""

from __future__ import annotations

import json
import re

from molsim_agent.agent.messages import Message
from molsim_agent.llm.base import LLMBackend


SUPPORTED_TARGETS = {"vasp", "xyz", "extxyz", "traj", "lammps-data", "cif"}


def rewrite_objective(backend: LLMBackend, objective: str) -> str | None:
    """Return a conservative canonical conversion request, or ``None``."""
    messages = [
        Message(
            role="system",
            content=(
                "You normalize molecular file-conversion requests. Return JSON only with "
                "keys source, destination, target_format, and validate. Use only target "
                "formats vasp, xyz, extxyz, traj, lammps-data, or cif. Never invent a "
                "source path, scientific parameter, unit, or workflow mapping. Set unknown "
                "values to null and validate to true only when validation is requested."
            ),
        ),
        Message(role="user", content=objective),
    ]
    response = backend.chat(messages, [])
    if response.tool_calls or not response.content.strip():
        return None
    candidate = response.content.strip()
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    source = data.get("source")
    destination = data.get("destination")
    target = data.get("target_format")
    validate = data.get("validate", False)
    if source is not None and not isinstance(source, str):
        return None
    if destination is not None and not isinstance(destination, str):
        return None
    if target is not None and (not isinstance(target, str) or target not in SUPPORTED_TARGETS):
        return None
    if not isinstance(validate, bool) or source is None or target is None:
        return None
    destination_text = destination or f"a {target} file"
    result = f"Convert {source} to {destination_text}"
    if validate:
        result += ", then validate the conversion"
    return result
