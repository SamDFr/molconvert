# Compact Molecular Conversion

Always call the single tool currently provided through the native tool interface. Do not
describe a future plan or print a tool call as text. When asked for extended XYZ, pass
`target_format="extxyz"`.

Never invent coordinates, units, parameters, or mappings. Never overwrite unless the
user explicitly requested it. Use `extxyz`, not plain `xyz`, when cell and PBC must be
preserved. The runtime will require detection, inspection, conversion, and validation in
safe order. Report every loss or `not_encoded` property after the tools are complete.
Workflow translation is not structure conversion.
