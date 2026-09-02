from pathlib import Path
from shutil import copyfile

from molsim_agent import Agent
from molsim_agent.agent.messages import LLMResponse, ToolCall
from molsim_agent.llm.mock import MockBackend


FIXTURE = Path(__file__).parent / "fixtures" / "POSCAR"


def test_mock_agent_completes_poscar_to_extxyz_milestone(tmp_path: Path) -> None:
    copyfile(FIXTURE, tmp_path / "POSCAR")
    backend = MockBackend(
        [
            LLMResponse(tool_calls=[ToolCall("1", "detect_file_format", {"path": "POSCAR"})]),
            LLMResponse(tool_calls=[ToolCall("2", "inspect_structure", {"path": "POSCAR"})]),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        "3",
                        "convert_structure",
                        {
                            "source": "POSCAR",
                            "destination": "structure.xyz",
                            "target_format": "extxyz",
                        },
                    )
                ]
            ),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        "4",
                        "validate_conversion",
                        {"source": "POSCAR", "destination": "structure.xyz"},
                    )
                ]
            ),
            LLMResponse(
                content=(
                    "Converted POSCAR to structure.xyz. Atom count, species, positions, "
                    "cell, and PBC were preserved."
                )
            ),
        ]
    )

    state = Agent(backend=backend, workspace=tmp_path).run(
        "Convert POSCAR to extended XYZ as structure.xyz, then validate it."
    )

    assert (tmp_path / "structure.xyz").is_file()
    assert state.iteration_count == 5
    assert state.created_files == ["structure.xyz"]
    validation = state.tool_executions[-1].result["result"]
    assert validation["required_structure_properties_preserved"] is True
    assert "were preserved" in state.final_answer


def test_runtime_requires_validation_after_conversion(tmp_path: Path) -> None:
    copyfile(FIXTURE, tmp_path / "POSCAR")
    backend = MockBackend(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall("detect", "detect_file_format", {"path": "POSCAR"})
                ]
            ),
            LLMResponse(
                tool_calls=[
                    ToolCall("inspect", "inspect_structure", {"path": "POSCAR"})
                ]
            ),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        "1",
                        "convert_structure",
                        {
                            "source": "POSCAR",
                            "destination": "structure.xyz",
                            "target_format": "extxyz",
                        },
                    )
                ]
            ),
            LLMResponse(content="Conversion complete."),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        "2",
                        "validate_conversion",
                        {"source": "POSCAR", "destination": "structure.xyz"},
                    )
                ]
            ),
            LLMResponse(content="Conversion and validation complete."),
        ]
    )

    state = Agent(backend=backend, workspace=tmp_path, profile="compact").run(
        "Convert and validate POSCAR"
    )

    assert state.iteration_count == 6
    assert state.final_answer == "Conversion and validation complete."
    assert any(
        "do not answer yet" in message.content for message in state.messages
    )
    exposed = [
        [schema["function"]["name"] for schema in request[1]]
        for request in backend.requests
    ]
    assert exposed == [
        ["detect_file_format"],
        ["inspect_structure"],
        ["convert_structure"],
        ["validate_conversion"],
        ["validate_conversion"],
        [],
    ]


def test_compact_inspection_does_not_send_coordinates(tmp_path: Path) -> None:
    copyfile(FIXTURE, tmp_path / "POSCAR")
    agent = Agent(
        backend=MockBackend([LLMResponse(content="Done")]),
        workspace=tmp_path,
        profile="compact",
    )

    result = agent.registry.execute("inspect_structure", {"path": "POSCAR"})

    assert "positions_angstrom" not in result["structure"]
    assert result["structure"]["atom_count"] == 2
