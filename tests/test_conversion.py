from pathlib import Path
from shutil import copyfile

import pytest

from molsim_agent.safety.policies import Workspace
from molsim_agent.tools.convert import convert_structure
from molsim_agent.tools.inspect import inspect_structure
from molsim_agent.tools.validate import validate_conversion


FIXTURE = Path(__file__).parent / "fixtures" / "POSCAR"


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    copyfile(FIXTURE, tmp_path / "POSCAR")
    return Workspace.from_path(tmp_path)


def test_poscar_to_extended_xyz_preserves_required_structure(workspace: Workspace) -> None:
    result = convert_structure(workspace, "POSCAR", "structure.xyz", "extxyz")
    report = validate_conversion(workspace, "POSCAR", "structure.xyz")

    assert result["target_format"] == "extxyz"
    assert result["conversion_fidelity"] == "exact_for_detected_properties_pending_validation"
    assert report["required_structure_properties_preserved"] is True
    assert report["atom_count"] == "preserved"
    assert report["species"] == "preserved"
    assert report["positions"]["status"] == "preserved"
    assert report["cell"]["status"] == "preserved"
    assert report["pbc"] == "preserved"


def test_poscar_to_lammps_data_preserves_basic_structure(workspace: Workspace) -> None:
    convert_structure(workspace, "POSCAR", "structure.data", "lammps-data")
    report = validate_conversion(workspace, "POSCAR", "structure.data")

    assert report["atom_count"] == "preserved"
    assert report["species"] == "preserved"
    assert report["positions"]["status"] == "preserved"
    assert report["cell"]["status"] == "preserved"
    assert report["pbc"] == "not_encoded"
    assert report["required_structure_properties_preserved"] is False


def test_poscar_to_cif_preserves_basic_structure(workspace: Workspace) -> None:
    result = convert_structure(workspace, "POSCAR", "structure.cif", "cif")
    report = validate_conversion(workspace, "POSCAR", "structure.cif")

    assert result["target_format"] == "cif"
    assert report["atom_count"] == "preserved"
    assert report["species"] == "preserved"


def test_xyz_to_ase_traj(workspace: Workspace) -> None:
    convert_structure(workspace, "POSCAR", "plain.xyz", "xyz")
    convert_structure(workspace, "plain.xyz", "structure.traj", "traj")

    inspected = inspect_structure(workspace, "structure.traj")
    assert inspected["structure"]["atom_count"] == 2


def test_plain_xyz_validation_reports_lost_cell_and_pbc(workspace: Workspace) -> None:
    conversion = convert_structure(workspace, "POSCAR", "plain.xyz", "xyz")
    report = validate_conversion(workspace, "POSCAR", "plain.xyz")

    assert conversion["conversion_fidelity"] == "potentially_lossy"
    assert conversion["detected_properties_at_risk"] == ["cell", "pbc"]
    assert report["cell"]["status"] == "lost"
    assert report["pbc"] == "lost"
    assert report["required_structure_properties_preserved"] is False


def test_conversion_refuses_overwrite_by_default(workspace: Workspace) -> None:
    convert_structure(workspace, "POSCAR", "structure.xyz", "extxyz")

    with pytest.raises(FileExistsError, match="overwrite=true"):
        convert_structure(workspace, "POSCAR", "structure.xyz", "extxyz")


def test_conversion_cannot_write_outside_workspace(workspace: Workspace) -> None:
    with pytest.raises(ValueError, match="outside"):
        convert_structure(workspace, "POSCAR", "../escaped.xyz", "extxyz")
