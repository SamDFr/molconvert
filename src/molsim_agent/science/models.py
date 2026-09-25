"""Typed, serialisable models used by scientific workflows.

The orchestrator may propose these values, but trusted tools must validate and
execute them.  Keeping validation here prevents free-form model output from
becoming an unchecked simulation configuration.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping


class ScientificSpecError(ValueError):
    """Raised when a scientific specification is physically or structurally invalid."""


class CapabilityStatus(StrEnum):
    """Why a scientific request can or cannot currently be executed."""

    AVAILABLE = "available"
    NEEDS_IMPLEMENTATION = "needs_implementation"
    NEEDS_DEPENDENCY = "needs_dependency"
    NEEDS_EXTERNAL_DATA = "needs_external_data"
    NEEDS_COMPUTE = "needs_compute"
    NEEDS_USER_INPUT = "needs_user_input"
    UNSUPPORTED = "unsupported"


@dataclass(slots=True)
class CapabilityAssessment:
    """Structured capability check, rather than an implicit prompt assertion."""

    status: CapabilityStatus | str
    task: str
    reason: str | None = None
    missing_capability: str | None = None
    proposed_solution: str | None = None
    delegate_to: str | None = None

    def __post_init__(self) -> None:
        try:
            self.status = CapabilityStatus(self.status)
        except ValueError as exc:
            raise ValueError(f"Unknown capability status: {self.status!r}") from exc
        if not self.task.strip():
            raise ValueError("CapabilityAssessment.task must not be empty")
        if self.status is CapabilityStatus.AVAILABLE and self.missing_capability:
            raise ValueError("An available capability cannot be missing")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "task": self.task,
            "reason": self.reason,
            "missing_capability": self.missing_capability,
            "proposed_solution": self.proposed_solution,
            "delegate_to": self.delegate_to,
        }


@dataclass(slots=True)
class ParameterValue:
    """A scientific parameter with provenance instead of an unexplained literal."""

    name: str
    value: Any = None
    source: str = "missing"  # user, default, derived, or missing
    requires_review: bool = True
    note: str | None = None

    def __post_init__(self) -> None:
        if self.source not in {"user", "default", "derived", "missing"}:
            raise ScientificSpecError("parameter source must be user, default, derived, or missing")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ScientificPlan:
    """Code-neutral plan separating defaults, facts, and unresolved inputs."""

    task: str
    code: str
    parameters: list[ParameterValue] = field(default_factory=list)
    missing_inputs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "code": self.code,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "missing_inputs": list(self.missing_inputs),
            "warnings": list(self.warnings),
            "artifacts": list(self.artifacts),
        }


@dataclass(slots=True)
class MDSpec:
    """A validated, backend-neutral molecular-dynamics protocol."""

    structure: str
    calculator: str
    ensemble: str
    timestep_fs: float
    steps: int
    temperature_K: float | None = None
    thermostat: str | None = None
    friction: float | None = None
    seed: int | None = None
    output_interval: int = 1

    def __post_init__(self) -> None:
        if not self.structure.strip():
            raise ScientificSpecError("structure must not be empty")
        if not self.calculator.strip():
            raise ScientificSpecError("calculator must not be empty")
        self.ensemble = self.ensemble.upper()
        if self.ensemble not in {"NVE", "NVT"}:
            raise ScientificSpecError("ensemble must be NVE or NVT")
        if self.timestep_fs <= 0:
            raise ScientificSpecError("timestep_fs must be positive")
        if self.steps <= 0:
            raise ScientificSpecError("steps must be positive")
        if self.output_interval <= 0:
            raise ScientificSpecError("output_interval must be positive")
        if self.temperature_K is not None and self.temperature_K <= 0:
            raise ScientificSpecError("temperature_K must be positive")
        if self.friction is not None and self.friction < 0:
            raise ScientificSpecError("friction must be non-negative")
        if self.ensemble == "NVT" and self.temperature_K is None:
            raise ScientificSpecError("NVT requires temperature_K")
        if self.seed is not None and self.seed < 0:
            raise ScientificSpecError("seed must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SinglePointSpec:
    structure: str
    calculator: str


@dataclass(slots=True)
class OptimizationSpec:
    structure: str
    calculator: str
    fmax: float = 0.05
    steps: int = 200

    def __post_init__(self) -> None:
        if not self.structure.strip() or not self.calculator.strip():
            raise ScientificSpecError("structure and calculator must not be empty")
        if self.fmax <= 0 or self.steps < 1:
            raise ScientificSpecError("fmax must be positive and steps must be positive")


@dataclass(slots=True)
class ExperimentRecord:
    """Reprovenance record written next to a scientific calculation."""

    id: str
    kind: str
    status: str
    input_structure: str
    input_hash: str | None = None
    potential: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    software_versions: dict[str, str] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in ("id", "kind", "status", "input_structure"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")

    @staticmethod
    def hash_file(path: str | Path) -> str:
        """Return a stable SHA-256 hash without loading a large file in memory."""
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @classmethod
    def for_input(
        cls,
        *,
        id: str,
        kind: str,
        status: str,
        input_structure: str | Path,
        **kwargs: Any,
    ) -> "ExperimentRecord":
        path = Path(input_structure)
        return cls(
            id=id,
            kind=kind,
            status=status,
            input_structure=str(input_structure),
            input_hash=cls.hash_file(path) if path.is_file() else None,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write_json(self, path: str | Path, *, indent: int = 2) -> None:
        Path(path).write_text(self.to_json(indent=indent) + "\n")

    @staticmethod
    def default_environment() -> dict[str, str]:
        """Minimal non-secret runtime provenance for new experiment records."""
        return {"python": sys.version.split()[0], "platform": platform.platform()}
