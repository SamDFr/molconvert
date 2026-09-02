"""Run a real compact-profile POSCAR-to-extXYZ conversion through Ollama."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from molsim_agent import Agent
from molsim_agent.cli import ConsoleEvents
from molsim_agent.llm.ollama import OllamaBackend, OllamaError


POSCAR = """H2 cubic demonstration
1.0
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
H
2
Cartesian
1.0 1.0 1.0
2.0 2.0 2.0
"""

OBJECTIVE = """Convert POSCAR to extended XYZ as structure.xyz. Then verify that atom
count, species, positions, cell, and PBC were preserved."""


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="granite3.3:2b")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--keep-workspace",
        type=Path,
        help="Use this directory and keep POSCAR/structure.xyz instead of a temporary one",
    )
    args = parser.parse_args()

    temporary = None
    if args.keep_workspace is None:
        temporary = tempfile.TemporaryDirectory(prefix="molsim-conversion-demo-")
        workspace = Path(temporary.name)
    else:
        workspace = args.keep_workspace.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)

    source = workspace / "POSCAR"
    destination = workspace / "structure.xyz"
    if source.exists() or destination.exists():
        print("FAILED: demo refuses to overwrite POSCAR or structure.xyz")
        if temporary is not None:
            temporary.cleanup()
        return 1
    source.write_text(POSCAR, encoding="utf-8")

    print("Live molecular conversion demo")
    print(f"Model: {args.model}")
    print(f"Profile: compact")
    print(f"Workspace: {workspace}")
    print(f"Objective: {OBJECTIVE}")

    agent = Agent(
        backend=OllamaBackend(args.model, timeout=args.timeout, think=False),
        workspace=workspace,
        profile="compact",
        max_iterations=8,
        event_handler=ConsoleEvents(verbose=False),
    )
    try:
        state = agent.run(OBJECTIVE)
    except OllamaError as exc:
        print(f"\nFAILED: {exc}")
        if temporary is not None:
            temporary.cleanup()
        return 1

    print(f"\nFinal:\n{state.final_answer}")
    if destination.is_file():
        print("\nGenerated structure.xyz:")
        print(destination.read_text(encoding="utf-8"))
    calls = [execution.call.name for execution in state.tool_executions]
    validation_ok = any(
        execution.call.name == "validate_conversion"
        and execution.result.get("ok")
        and execution.result["result"].get("required_structure_properties_preserved")
        for execution in state.tool_executions
    )
    passed = destination.is_file() and validation_ok
    print(f"Tool calls: {calls}")
    print("LIVE CONVERSION PASSED" if passed else "LIVE CONVERSION FAILED")
    if temporary is not None:
        temporary.cleanup()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
