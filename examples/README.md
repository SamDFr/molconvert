# Example workspace

Copy `tests/fixtures/POSCAR` here, then run from the repository root:

```bash
molsim-agent --workspace examples --model qwen3:8b
```

Prompt:

```text
Convert POSCAR to extended XYZ as structure.xyz. Then verify that atom count,
species, positions, cell and PBC were preserved.
```
