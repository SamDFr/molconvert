"""Filesystem containment rules shared by all tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class SafetyError(ValueError):
    """Raised when a requested operation violates agent policy."""


@dataclass(frozen=True, slots=True)
class Workspace:
    root: Path

    @classmethod
    def from_path(cls, path: str | Path) -> Workspace:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise SafetyError(f"Workspace is not a directory: {root}")
        return cls(root=root)

    def resolve(self, path: str | Path, *, must_exist: bool = False) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(self.root):
            raise SafetyError(f"Path is outside the workspace: {path}")
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        return resolved

    def relative(self, path: Path) -> str:
        return str(path.relative_to(self.root)) or "."
