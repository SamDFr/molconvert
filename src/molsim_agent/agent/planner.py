"""Instruction assembly for the model-facing planning context."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def default_skill_path(profile: str = "full") -> Path:
    """Locate the repository's built-in skill in a source or editable install."""

    skill_name = "molecular-conversion-compact" if profile == "compact" else "molecular-conversion"
    repository_path = (
        Path(__file__).resolve().parents[3]
        / "skills"
        / skill_name
        / "SKILL.md"
    )
    if repository_path.is_file():
        return repository_path
    return (
        Path(__file__).resolve().parents[1]
        / "skills"
        / skill_name
        / "SKILL.md"
    )


def build_system_prompt(
    base_prompt: str,
    skill_paths: Sequence[str | Path] | None = None,
    *,
    profile: str = "full",
) -> str:
    paths = (
        [Path(path) for path in skill_paths]
        if skill_paths is not None
        else [default_skill_path(profile)]
    )
    sections = [base_prompt.strip()]
    for path in paths:
        if not path.is_file():
            if skill_paths is not None:
                raise FileNotFoundError(f"Skill file does not exist: {path}")
            continue
        sections.append(f"Loaded skill instructions from {path.name}:\n\n{path.read_text(encoding='utf-8').strip()}")
    return "\n\n---\n\n".join(sections)
