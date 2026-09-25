"""Workflow descriptors kept separate from the generic Agent loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Workflow:
    name: str
    description: str
    allowed_tools: tuple[str, ...] = ()
    completion_requirements: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversionWorkflow(Workflow):
    def __init__(self) -> None:
        super().__init__(
            "conversion",
            "Deterministic structure conversion followed by validation.",
            ("detect_file_format", "inspect_structure", "convert_structure", "validate_conversion"),
            ("detect_file_format", "inspect_structure", "convert_structure", "validate_conversion"),
        )


class ResearchWorkflow(Workflow):
    def __init__(self) -> None:
        super().__init__("research", "Scientific planning, execution, and interpretation.")
