"""Validation boundary for temporary ScientificCodeAgent artifacts."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def validate_generated_tool(directory: str | Path, timeout: int = 30) -> dict[str, object]:
    root = Path(directory).resolve()
    if not root.is_dir() or not (root / "tool.py").is_file():
        raise FileNotFoundError("generated tool directory must contain tool.py")
    if timeout < 1:
        raise ValueError("timeout must be positive")
    test = root / "test_tool.py"
    command = [sys.executable, "-m", "pytest", "-q", str(test)] if test.exists() else [sys.executable, "-m", "py_compile", str(root / "tool.py")]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=timeout)
    metadata = {"ok": completed.returncode == 0, "returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]}
    (root / "validation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
