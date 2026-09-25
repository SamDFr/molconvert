"""Trusted, small ASE simulation tools.

These functions are deliberately boring: an LLM may propose a structured
protocol, but ASE performs the calculation and this module records exactly what
was executed.  Heavy calculators remain optional through :class:`PotentialRegistry`.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from ase import units

from molsim_agent.formats.structures import read_structure
from molsim_agent.potentials.registry import PotentialRegistry
from molsim_agent.safety.policies import Workspace
from molsim_agent.science.models import ExperimentRecord, MDSpec, OptimizationSpec, SinglePointSpec
from molsim_agent.simulation.compare import compare_single_point
from molsim_agent.tools.registry import ToolSpec


def _spec_dict(spec: Any) -> dict[str, Any]:
    if is_dataclass(spec):
        return asdict(spec)
    if isinstance(spec, dict):
        return dict(spec)
    raise TypeError("spec must be a dataclass specification or JSON object")


def _run_directory(workspace: Workspace, kind: str, output_dir: str | None) -> tuple[Path, str]:
    if output_dir:
        path = workspace.resolve(output_dir)
        if path.exists() and not path.is_dir():
            raise FileExistsError(f"Simulation output path is not a directory: {output_dir}")
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(
                f"Simulation output directory is not empty: {output_dir}; "
                "choose a new run directory to protect existing results"
            )
        path.mkdir(parents=True, exist_ok=True)
        run_id = path.name
    else:
        run_id = f"{kind}-{uuid.uuid4().hex[:10]}"
        path = workspace.resolve(Path("runs") / run_id)
        path.mkdir(parents=True, exist_ok=False)
    return path, run_id


def _load_spec(cls: type[Any], value: Any) -> Any:
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        return cls(**value)
    raise TypeError(f"spec must be {cls.__name__} or an object")


def _calculator(name: str, config: dict[str, Any] | None = None):
    return PotentialRegistry().create_calculator(name, config or {})


def _record(workspace: Workspace, run_id: str, kind: str, source: Path, status: str,
            parameters: dict[str, Any], outputs: list[str], warnings: list[str] | None = None) -> dict[str, Any]:
    record = ExperimentRecord(
        id=run_id,
        kind=kind,
        status=status,
        input_structure=workspace.relative(source),
        input_hash=ExperimentRecord.hash_file(source) if source.is_file() else None,
        potential={"name": parameters.get("calculator", parameters.get("potential", "unknown"))},
        parameters=parameters,
        software_versions={"ase": _ase_version()},
        environment=ExperimentRecord.default_environment(),
        outputs=outputs,
        warnings=warnings or [],
    )
    return record


def _ase_version() -> str:
    try:
        import ase
        return str(ase.__version__)
    except Exception:
        return "unknown"


def _write_record(record: ExperimentRecord, run_path: Path) -> str:
    record_path = run_path / "experiment.json"
    record.write_json(record_path)
    return record_path.name


def single_point(workspace: Workspace, spec: SinglePointSpec | dict[str, Any], output_dir: str | None = None) -> dict[str, Any]:
    """Evaluate energy and forces for one structure with a registered calculator."""
    protocol = _load_spec(SinglePointSpec, spec)
    source = workspace.resolve(protocol.structure, must_exist=True)
    atoms, source_format, read_warnings = read_structure(source)
    atoms.calc = _calculator(protocol.calculator)
    run_path, run_id = _run_directory(workspace, "single-point", output_dir)
    try:
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces()).tolist()
        results = {"energy_eV": energy, "forces_eV_per_angstrom": forces,
                   "atom_count": len(atoms), "source_format": source_format}
        (run_path / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        record = _record(workspace, run_id, "single_point", source, "success", _spec_dict(protocol), ["results.json"], read_warnings)
        record_name = _write_record(record, run_path)
        return {"ok": True, "run_id": run_id, "output_dir": workspace.relative(run_path),
                "results": results, "experiment_record": workspace.relative(run_path / record_name),
                "warnings": read_warnings}
    except Exception as exc:
        record = _record(workspace, run_id, "single_point", source, "failed", _spec_dict(protocol), [], [str(exc)])
        _write_record(record, run_path)
        raise


def geometry_optimization(workspace: Workspace, spec: OptimizationSpec | dict[str, Any], output_dir: str | None = None) -> dict[str, Any]:
    """Run a bounded BFGS geometry optimisation and save trajectory/output."""
    protocol = _load_spec(OptimizationSpec, spec)
    source = workspace.resolve(protocol.structure, must_exist=True)
    atoms, source_format, read_warnings = read_structure(source)
    atoms.calc = _calculator(protocol.calculator)
    run_path, run_id = _run_directory(workspace, "optimization", output_dir)
    trajectory = run_path / "optimization.traj"
    optimizer = BFGS(atoms, logfile=str(run_path / "optimization.log"), trajectory=str(trajectory))
    try:
        optimizer.run(fmax=protocol.fmax, steps=protocol.steps)
        output = run_path / "optimized.extxyz"
        write(output, atoms, format="extxyz")
        results = {"converged": bool(optimizer.converged()), "steps": int(optimizer.nsteps),
                   "final_energy_eV": float(atoms.get_potential_energy()), "source_format": source_format}
        (run_path / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        outputs = ["optimized.extxyz", "optimization.traj", "optimization.log", "results.json"]
        record = _record(workspace, run_id, "geometry_optimization", source, "success", _spec_dict(protocol), outputs, read_warnings)
        _write_record(record, run_path)
        return {"ok": True, "run_id": run_id, "output_dir": workspace.relative(run_path), "results": results,
                "outputs": outputs, "warnings": read_warnings}
    except Exception as exc:
        record = _record(workspace, run_id, "geometry_optimization", source, "failed", _spec_dict(protocol), [], [str(exc)])
        _write_record(record, run_path)
        raise


def run_md(workspace: Workspace, spec: MDSpec | dict[str, Any], output_dir: str | None = None) -> dict[str, Any]:
    """Run a small NVE or Langevin NVT trajectory with a registered calculator."""
    protocol = _load_spec(MDSpec, spec)
    source = workspace.resolve(protocol.structure, must_exist=True)
    atoms, source_format, read_warnings = read_structure(source)
    atoms.calc = _calculator(protocol.calculator)
    run_path, run_id = _run_directory(workspace, "md", output_dir)
    trajectory = run_path / "trajectory.traj"
    rng = np.random.default_rng(protocol.seed)
    if protocol.temperature_K is not None and not atoms.has("momenta"):
        MaxwellBoltzmannDistribution(atoms, temperature_K=protocol.temperature_K, rng=rng)
    if protocol.ensemble == "NVT":
        dynamics = Langevin(atoms, protocol.timestep_fs * units.fs, temperature_K=protocol.temperature_K,
                            friction=protocol.friction if protocol.friction is not None else 0.01)
    else:
        dynamics = VelocityVerlet(atoms, protocol.timestep_fs * units.fs)
    dynamics.attach(lambda: write(trajectory, atoms, format="traj", append=True), interval=protocol.output_interval)
    try:
        dynamics.run(protocol.steps)
        results = {"ensemble": protocol.ensemble, "steps": protocol.steps,
                   "frames_written": (protocol.steps // protocol.output_interval) + 1,
                   "final_temperature_K": float(atoms.get_temperature()),
                   "source_format": source_format}
        (run_path / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        outputs = ["trajectory.traj", "results.json"]
        record = _record(workspace, run_id, "molecular_dynamics", source, "success", _spec_dict(protocol), outputs, read_warnings)
        _write_record(record, run_path)
        return {"ok": True, "run_id": run_id, "output_dir": workspace.relative(run_path), "results": results,
                "outputs": outputs, "warnings": read_warnings}
    except Exception as exc:
        record = _record(workspace, run_id, "molecular_dynamics", source, "failed", _spec_dict(protocol), [], [str(exc)])
        _write_record(record, run_path)
        raise


def simulation_tool_specs(workspace: Workspace) -> list[ToolSpec]:
    """Model-visible specifications for the trusted simulation functions."""
    spec_schema = {"type": "object", "additionalProperties": True}
    return [
        ToolSpec("single_point", "Run a deterministic single-point energy/force calculation.",
                 {"type": "object", "properties": {"spec": spec_schema, "output_dir": {"type": "string"}}, "required": ["spec"], "additionalProperties": False},
                 lambda spec, output_dir=None: single_point(workspace, spec, output_dir), category="simulation", risk="compute", requirements=("ase",), compute_cost="cpu"),
        ToolSpec("geometry_optimization", "Optimize a structure with a validated ASE protocol.",
                 {"type": "object", "properties": {"spec": spec_schema, "output_dir": {"type": "string"}}, "required": ["spec"], "additionalProperties": False},
                 lambda spec, output_dir=None: geometry_optimization(workspace, spec, output_dir), category="simulation", risk="compute", requirements=("ase",), compute_cost="cpu"),
        ToolSpec("run_md", "Run a bounded NVE or NVT molecular-dynamics protocol.",
                 {"type": "object", "properties": {"spec": spec_schema, "output_dir": {"type": "string"}}, "required": ["spec"], "additionalProperties": False},
                 lambda spec, output_dir=None: run_md(workspace, spec, output_dir), category="simulation", risk="compute", requirements=("ase",), compute_cost="cpu", deterministic=False),
        ToolSpec("compare_single_point", "Evaluate identical fixed configurations with multiple calculators.",
                 {"type": "object", "properties": {"structure": {"type": "string"}, "calculators": {"type": "array", "items": {"type": "string"}}}, "required": ["structure", "calculators"], "additionalProperties": False},
                 lambda structure, calculators: compare_single_point(workspace, structure, calculators), category="analysis", risk="compute", requirements=("ase",), compute_cost="cpu", deterministic=True),
    ]
