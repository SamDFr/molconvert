from __future__ import annotations

from molsim_agent import Agent
from molsim_agent.agent.messages import LLMResponse, ToolCall
from molsim_agent.agent.state import AgentState
from molsim_agent.llm.mock import MockBackend
from molsim_agent.llm.ollama import OllamaBackend


def test_agent_executes_tool_and_returns_observation_to_model(tmp_path) -> None:
    (tmp_path / "POSCAR").write_text("fixture", encoding="utf-8")
    backend = MockBackend(
        [
            LLMResponse(
                tool_calls=[ToolCall(id="call-1", name="list_directory", arguments={"path": "."})]
            ),
            LLMResponse(content="I found POSCAR."),
        ]
    )

    state = Agent(backend=backend, workspace=tmp_path).run("What files are here?")

    assert state.final_answer == "I found POSCAR."
    assert state.iteration_count == 2
    assert state.tool_executions[0].result["ok"] is True
    assert "POSCAR" in backend.requests[1][0][-1].content
    assert backend.requests[0][1][0]["function"]["name"] == "list_directory"


def test_agent_turns_tool_failure_into_an_observation(tmp_path) -> None:
    backend = MockBackend(
        [
            LLMResponse(
                tool_calls=[ToolCall(id="bad", name="missing_tool", arguments={})]
            ),
            LLMResponse(content="The requested tool was unavailable."),
        ]
    )

    state = Agent(backend=backend, workspace=tmp_path).run("Do something")

    assert state.tool_executions[0].result == {
        "ok": False,
        "error": "ValueError",
        "message": "Unknown tool: missing_tool",
    }
    assert state.final_answer == "The requested tool was unavailable."


def test_agent_stops_at_iteration_limit(tmp_path) -> None:
    backend = MockBackend(
        [LLMResponse(tool_calls=[ToolCall(str(i), "list_directory", {})]) for i in range(2)]
    )

    state = Agent(backend=backend, workspace=tmp_path, max_iterations=2).run("Keep looking")

    assert state.iteration_count == 2
    assert "maximum" in state.warnings[0]


def test_agent_loads_molecular_conversion_skill(tmp_path) -> None:
    backend = MockBackend([LLMResponse(content="Done")])

    Agent(backend=backend, workspace=tmp_path).run("Explain policy")

    system_message = backend.requests[0][0][0]
    assert "Molecular Structure Conversion" in system_message.content
    assert "Never invent or manually reproduce atomic coordinates" in system_message.content


def test_auto_profile_uses_compact_context_for_ollama(tmp_path) -> None:
    agent = Agent(
        backend=OllamaBackend("unused"), workspace=tmp_path, profile="auto"
    )

    assert agent.profile == "compact"
    assert {schema["function"]["name"] for schema in agent.registry.schemas()} == {
        "detect_file_format",
        "inspect_structure",
        "convert_structure",
        "validate_conversion",
    }
    assert "Compact Molecular Conversion" in agent.system_prompt


def test_progress_mode_requests_brief_model_status(tmp_path) -> None:
    backend = MockBackend([LLMResponse(content="Done")])
    agent = Agent(backend=backend, workspace=tmp_path, profile="compact", progress=True)

    message = agent._messages_for_model(
        AgentState(objective="Convert POSCAR to extended XYZ as structure.xyz")
    )[0]

    assert "progress sentence" in message.content


def test_detailed_progress_requests_factual_observations(tmp_path) -> None:
    backend = MockBackend([LLMResponse(content="Done")])
    agent = Agent(
        backend=backend, workspace=tmp_path, profile="compact", progress_level="detailed"
    )
    message = agent._messages_for_model(
        AgentState(objective="Convert POSCAR to extended XYZ as structure.xyz")
    )[0]

    assert "atom count" in message.content
    assert "never guess" in message.content


def test_progress_mode_requires_model_announcement_before_tool(tmp_path) -> None:
    backend = MockBackend(
        [
            LLMResponse(tool_calls=[ToolCall("1", "list_directory", {"path": "."})]),
            LLMResponse(
                content="I will inspect the workspace now.",
                tool_calls=[ToolCall("2", "list_directory", {"path": "."})],
            ),
            LLMResponse(content="The workspace is ready."),
        ]
    )
    state = Agent(
        backend=backend, workspace=tmp_path, progress_level="brief"
    ).run("List the files")

    assert len(state.tool_executions) == 1
    assert state.final_answer == "The workspace is ready."


def test_compact_constraints_resolve_find_poscar_convert_it(tmp_path) -> None:
    (tmp_path / "POSCAR").write_text("fixture", encoding="utf-8")
    agent = Agent(backend=MockBackend([]), workspace=tmp_path, profile="compact")
    state = AgentState(objective="Find the POSCAR and convert it to POSCAR.xyz")

    assert agent._compact_expected_arguments(state, "detect_file_format") == {"path": "POSCAR"}
    assert agent._compact_expected_arguments(state, "convert_structure") == {
        "source": "POSCAR",
        "destination": "POSCAR.xyz",
        "target_format": "xyz",
    }
