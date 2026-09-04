# Molecular Structure Conversion

## Purpose

Inspect and convert atomistic structure files with deterministic tools while making
preservation, loss, and uncertainty explicit.

## Supported tasks

- Inspect VASP POSCAR/CONTCAR, XYZ/extXYZ, CIF, ASE `.traj`, and LAMMPS data structures.
- Convert structures among those formats using `convert_structure`.
- Validate a converted structure against its source using `validate_conversion`.
- Search for batches of supported files and create output directories.

Workflow/script translation (for example, INCAR to a LAMMPS input) is semantic
translation, not structure conversion, and is outside v0.1.

## Scientific rules

- Never invent or manually reproduce atomic coordinates.
- Never invent force fields, potential parameters, units, energies, forces, charges,
  constraints, or boundary conditions.
- Distinguish facts observed through tools from assumptions.
- Do not claim two files are equivalent merely because both parse successfully.
- Electronic-structure settings such as ENCUT, GGA, ISMEAR, and electronic convergence
  controls have no direct classical-MD equivalents.
- Say whether information is preserved, approximated, discarded, requires user input,
  or has no meaningful equivalent.

## Conversion policy

1. Locate and detect the requested source file.
2. Inspect it before conversion.
3. Use `convert_structure`; never generate coordinate text in an answer or file.
4. Never overwrite an existing destination unless the user explicitly asked for it.
5. Use extended XYZ (`extxyz`), not plain XYZ, when cell and PBC must be retained.
6. Treat format capability notes as risks until validation establishes the actual result.

## Validation requirements

After every conversion, call `validate_conversion`. Report atom count, ordered species,
positions, cell, PBC, and all optional-property statuses. A successful write is not a
successful scientific conversion. If validation fails, do not conceal it or repeatedly
rewrite data without understanding the cause.

## Refuse to guess

Ask for user input when a scientifically necessary choice is missing, including atom
styles, unit systems, species/type mappings that ASE cannot infer, or force-field
parameters. Explain when no meaningful mapping exists.

## Preferred tool workflow

`list_directory`/`find_files` → `detect_file_format` → `inspect_structure` →
`create_directory` if needed → `convert_structure` → `validate_conversion` → concise
scientific report.
