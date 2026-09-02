"""Constrained filesystem tools."""

from __future__ import annotations

from pathlib import Path

from molsim_agent.safety.policies import Workspace
from molsim_agent.tools.registry import ToolSpec


def list_directory(workspace: Workspace, path: str = ".") -> dict[str, object]:
    directory = workspace.resolve(path, must_exist=True)
    if not directory.is_dir():
        raise NotADirectoryError(path)
    entries = [
        {
            "name": item.name,
            "path": workspace.relative(item),
            "type": "directory" if item.is_dir() else "file",
        }
        for item in sorted(directory.iterdir(), key=lambda item: item.name)
    ]
    return {"path": workspace.relative(directory), "entries": entries}


def find_files(workspace: Workspace, pattern: str, path: str = ".") -> dict[str, object]:
    directory = workspace.resolve(path, must_exist=True)
    if not directory.is_dir():
        raise NotADirectoryError(path)
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ValueError("Search pattern must be a relative glob without '..'")
    matches = [
        workspace.relative(item)
        for item in sorted(directory.glob(pattern))
        if item.is_file() and item.resolve().is_relative_to(workspace.root)
    ]
    return {"path": workspace.relative(directory), "pattern": pattern, "matches": matches}


def read_text_file(
    workspace: Workspace, path: str, max_chars: int = 100_000
) -> dict[str, object]:
    source = workspace.resolve(path, must_exist=True)
    if not source.is_file():
        raise ValueError(f"Not a file: {path}")
    if max_chars < 1 or max_chars > 1_000_000:
        raise ValueError("max_chars must be between 1 and 1000000")
    with source.open("r", encoding="utf-8") as handle:
        text = handle.read(max_chars + 1)
    truncated = len(text) > max_chars
    return {
        "path": workspace.relative(source),
        "content": text[:max_chars],
        "truncated": truncated,
        "characters_returned": min(len(text), max_chars),
    }


def create_directory(workspace: Workspace, path: str) -> dict[str, object]:
    destination = workspace.resolve(path)
    existed = destination.exists()
    if existed and not destination.is_dir():
        raise FileExistsError(f"A non-directory already exists at: {path}")
    destination.mkdir(parents=True, exist_ok=True)
    return {
        "path": workspace.relative(destination),
        "created": not existed,
        "created_files": [workspace.relative(destination)] if not existed else [],
    }


def filesystem_tool_specs(workspace: Workspace) -> list[ToolSpec]:
    object_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative path."}
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    return [
        ToolSpec(
            name="list_directory",
            description="List files and subdirectories at a path inside the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative directory path; defaults to '.'.",
                        "default": ".",
                    }
                },
                "additionalProperties": False,
            },
            function=lambda path=".": list_directory(workspace, path),
            safety={"filesystem": "read", "workspace_only": True},
        ),
        ToolSpec(
            name="find_files",
            description="Find files recursively or non-recursively with a glob such as '**/POSCAR'.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
            function=lambda pattern, path=".": find_files(workspace, pattern, path),
            safety={"filesystem": "read", "workspace_only": True},
        ),
        ToolSpec(
            name="read_text_file",
            description="Read a UTF-8 text file inside the workspace with a size limit.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer", "minimum": 1, "maximum": 1000000},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            function=lambda path, max_chars=100_000: read_text_file(workspace, path, max_chars),
            safety={"filesystem": "read", "workspace_only": True},
        ),
        ToolSpec(
            name="create_directory",
            description="Create a directory inside the workspace, including missing parents.",
            parameters=object_schema,
            function=lambda path: create_directory(workspace, path),
            safety={"filesystem": "write", "workspace_only": True},
        ),
    ]
