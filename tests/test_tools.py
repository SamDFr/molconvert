from pathlib import Path

import pytest

from molsim_agent.safety.policies import SafetyError, Workspace
from molsim_agent.tools.filesystem import create_directory, find_files, list_directory
from molsim_agent.tools.registry import ToolRegistry, ToolSpec


def test_filesystem_tools_are_confined_to_workspace(tmp_path: Path) -> None:
    workspace = Workspace.from_path(tmp_path)

    with pytest.raises(SafetyError, match="outside"):
        list_directory(workspace, "..")
    with pytest.raises(SafetyError, match="outside"):
        create_directory(workspace, "../escaped")


def test_find_files_supports_recursive_glob(tmp_path: Path) -> None:
    nested = tmp_path / "runs" / "one"
    nested.mkdir(parents=True)
    (nested / "POSCAR").write_text("data", encoding="utf-8")

    result = find_files(Workspace.from_path(tmp_path), "**/POSCAR", "runs")

    assert result["matches"] == ["runs/one/POSCAR"]


def test_registry_rejects_wrong_argument_type() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "example",
            "Example",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            lambda path: {"path": path},
        )
    )

    with pytest.raises(TypeError, match="JSON type string"):
        registry.execute("example", {"path": 42})
