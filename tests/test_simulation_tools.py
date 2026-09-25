import json

from ase import Atoms
from ase.io import write, read

from molsim_agent.safety.policies import Workspace
from molsim_agent.science import MDSpec, OptimizationSpec, SinglePointSpec
from molsim_agent.tools.simulation import geometry_optimization, run_md, single_point


def _workspace(tmp_path):
    source = tmp_path / "h2.xyz"
    write(source, Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.9]]), format="xyz")
    return Workspace.from_path(tmp_path), source


def test_single_point_records_energy_and_provenance(tmp_path):
    workspace, source = _workspace(tmp_path)
    result = single_point(workspace, SinglePointSpec(str(source), "emt"))
    assert result["ok"] is True
    run = tmp_path / result["output_dir"]
    payload = json.loads((run / "results.json").read_text())
    assert "energy_eV" in payload and len(payload["forces_eV_per_angstrom"]) == 2
    assert json.loads((run / "experiment.json").read_text())["input_hash"]


def test_optimization_writes_run_artifacts(tmp_path):
    workspace, source = _workspace(tmp_path)
    result = geometry_optimization(workspace, OptimizationSpec(str(source), "emt", steps=2))
    run = tmp_path / result["output_dir"]
    assert (run / "optimized.extxyz").exists()
    assert (run / "optimization.traj").exists()
    assert len(read(run / "optimized.extxyz")) == 2


def test_md_writes_trajectory_and_rejects_outside_output(tmp_path):
    workspace, source = _workspace(tmp_path)
    spec = MDSpec(str(source), "emt", "NVE", timestep_fs=0.1, steps=3, output_interval=1, seed=3)
    result = run_md(workspace, spec)
    run = tmp_path / result["output_dir"]
    assert (run / "trajectory.traj").exists()
    assert result["results"]["steps"] == 3
    try:
        run_md(workspace, spec, "../outside")
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("outside output directory was accepted")
