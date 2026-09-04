"""Conservative format detection for the formats supported in v0.1."""

from __future__ import annotations

from pathlib import Path


SUPPORTED_FORMATS = ("vasp", "xyz", "extxyz", "traj", "lammps-data", "cif")


def detect_format(path: Path) -> tuple[str | None, list[str]]:
    """Return a supported ASE format name and any uncertainty warnings."""

    name = path.name.lower()
    suffix = path.suffix.lower()
    if name in {"poscar", "contcar"} or suffix in {".vasp", ".poscar"}:
        return "vasp", []
    if suffix == ".traj":
        return "traj", []
    if suffix == ".cif":
        return "cif", []
    if suffix in {".data", ".lammps", ".lmp"} or "lammps" in name:
        return "lammps-data", []
    if suffix == ".xyz":
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.readline()
                comment = handle.readline()
        except OSError:
            return "xyz", ["Could not inspect the XYZ comment line"]
        if "Properties=" in comment or "Lattice=" in comment or "pbc=" in comment:
            return "extxyz", []
        return "xyz", [
            "Plain XYZ does not encode cell or periodic boundary conditions"
        ]
    return None, ["File extension/name is not recognized as a supported v0.1 format"]


def normalize_format(format_name: str) -> str:
    normalized = format_name.strip().lower().replace("_", "-")
    aliases = {
        "extended-xyz": "extxyz",
        "extendedxyz": "extxyz",
        "ase-traj": "traj",
        "lammps": "lammps-data",
        "lammpsdata": "lammps-data",
        "cif": "cif",
        "poscar": "vasp",
        "contcar": "vasp",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format {format_name!r}; supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )
    return normalized
