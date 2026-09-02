"""Small live Ollama test for resource-constrained computers.

This intentionally exposes one tool and short instructions. It proves that the real
model can request a Python tool, receive its observation, and produce a final answer.
It does not test molecular conversion; the deterministic suite covers that separately.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from molsim_agent import Agent
from molsim_agent.cli import ConsoleEvents
from molsim_agent.llm.ollama import OllamaBackend, OllamaError
from molsim_agent.safety.policies import Workspace
from molsim_agent.tools.filesystem import list_directory
from molsim_agent.tools.registry import ToolRegistry, ToolSpec


def compact_registry(workspace: Workspace) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="list_directory",
            description="List files in a workspace directory.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Use '.' for the root."}
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            function=lambda path: list_directory(workspace, path),
            safety={"filesystem": "read", "workspace_only": True},
        )
    )
    return registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="granite3.3:2b")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="molsim-ollama-smoke-") as temporary:
        root = Path(temporary)
        (root / "smoke_marker.txt").write_text("Ollama tool calling works.\n", encoding="utf-8")
        workspace = Workspace.from_path(root)
        backend = OllamaBackend(model=args.model, timeout=args.timeout)
        agent = Agent(
            backend=backend,
            workspace=root,
            registry=compact_registry(workspace),
            system_prompt=(
                "You are testing native tool calling. Call list_directory exactly once with "
                "path '.'. After its observation, answer whether smoke_marker.txt exists."
            ),
            skill_paths=[],
            max_iterations=3,
            event_handler=ConsoleEvents(verbose=False),
        )

        print("Ollama native tool-call smoke test")
        print(f"Model: {args.model}")
        print(f"Temporary workspace: {root}")
        try:
            state = agent.run("Inspect the workspace using the tool, then report what you found.")
        except OllamaError as exc:
            print(f"\nFAILED: {exc}")
            return 1

        print(f"\nFinal:\n{state.final_answer}")
        calls = [execution.call.name for execution in state.tool_executions]
        passed = calls == ["list_directory"] and state.final_answer is not None
        print(f"\nTool calls: {calls}")
        print("SMOKE TEST PASSED" if passed else "SMOKE TEST FAILED")
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
