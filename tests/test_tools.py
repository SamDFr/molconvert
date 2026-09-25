from pathlib import Path

import pytest

from molsim_agent.safety.policies import SafetyError, Workspace
from molsim_agent.tools.filesystem import create_directory, find_files, list_directory
from molsim_agent.tools.registry import ToolRegistry, ToolSpec
from molsim_agent.tools.gromacs import prepare_gromacs_md_inputs
from molsim_agent.tools.templates import template_arguments


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


def test_gromacs_template_writes_geometry_mdp_and_explicit_topology_gap(tmp_path: Path) -> None:
    (tmp_path / "POSCAR").write_text(
        "H2\n1.0\n5 0 0\n0 5 0\n0 0 5\nH\n2\nDirect\n0 0 0\n0.1 0.1 0.1\n",
        encoding="utf-8",
    )
    result = prepare_gromacs_md_inputs(Workspace.from_path(tmp_path))
    assert result["ok"] is True
    assert (tmp_path / "outputs" / "POSCAR.gro").exists()
    assert (tmp_path / "outputs" / "md.mdp").exists()
    assert (tmp_path / "outputs" / "topol.top.template").exists()
    assert any("force field" in warning for warning in result["warnings"])


def test_template_arguments_preserve_explicit_protocol_values() -> None:
    args = template_arguments("prepare LAMMPS MD in NVT at 10 K for 2 ps with timestep 0.5 fs")
    assert args == {
        "source": "POSCAR",
        "temperature_K": 10.0,
        "duration_ps": 2.0,
        "timestep_fs": 0.5,
        "ensemble": "nvt",
    }
