"""Workflow policies kept separate from the generic agent loop."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


@dataclass(slots=True)
class Workflow:
    name: str
    description: str
    allowed_tools: tuple[str, ...] = ()
    completion_requirements: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def matches(self, objective: str) -> bool:
        return False

    def next_tool(self, successful_tools: set[str]) -> str | None:
        return next((name for name in self.allowed_tools if name not in successful_tools), None)


class ConversionWorkflow(Workflow):
    def __init__(self) -> None:
        super().__init__(
            "conversion",
            "Deterministic structure conversion followed by validation.",
            ("detect_file_format", "inspect_structure", "convert_structure", "validate_conversion"),
            ("detect_file_format", "inspect_structure", "convert_structure", "validate_conversion"),
        )

    def matches(self, objective: str) -> bool:
        if re.search(r"\b(?:convert\w*|con\w*vert\w*|transform\w*|export\w*|write|save|turn|change|convertir|transformer)\b", objective, re.IGNORECASE):
            return True
        return bool(re.search(r"\b(?:to|into|as|en|vers|verso)\b.*\b(?:xyz|extxyz|cif|traj|lammps|poscar|structure|format)\b", objective, re.IGNORECASE | re.DOTALL))


class ResearchWorkflow(Workflow):
    def __init__(self) -> None:
        super().__init__("research", "Scientific planning, execution, and interpretation.")

    def matches(self, objective: str) -> bool:
        return True
