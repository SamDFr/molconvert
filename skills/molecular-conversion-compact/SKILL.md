# Compact Molecular Conversion

Always call the single tool currently provided through the native tool interface. Do not
describe a future plan or print a tool call as text. When asked for extended XYZ, pass
`target_format="extxyz"`.

Supported structure targets include VASP, plain XYZ, extended XYZ, CIF, ASE traj, and
LAMMPS data. Prefer an explicit destination filename; if it is omitted, use the
runtime-provided safe derived filename (for example `POSCAR.data`) rather than inventing
scientific content.

Never invent coordinates, units, parameters, or mappings. Never overwrite unless the
user explicitly requested it. If the requested output exists, preserve it and use a
numbered sibling (`file_1.ext`, etc.), reporting the actual destination. Use `extxyz`, not plain `xyz`, when cell and PBC must be
preserved. The runtime will require detection, inspection, conversion, and validation in
safe order. Report every loss or `not_encoded` property after the tools are complete.
Workflow translation is not structure conversion.
