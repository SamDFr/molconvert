from molsim_agent.agent.runtime import AgentRuntime
from molsim_agent.agent.workflows import ConversionWorkflow, ResearchWorkflow


def test_runtime_dispatches_conversion_objectives() -> None:
    runtime = AgentRuntime()
    assert runtime.workflow_for("convert POSCAR to XYZ").name == "conversion"


def test_runtime_falls_back_to_research_workflow() -> None:
    runtime = AgentRuntime()
    assert runtime.workflow_for("calculate an RDF").name == "research"


def test_conversion_workflow_exposes_ordered_completion_policy() -> None:
    workflow = ConversionWorkflow()
    assert workflow.next_tool(set()) == "detect_file_format"
    assert workflow.next_tool({"detect_file_format", "inspect_structure"}) == "convert_structure"
    assert workflow.next_tool(set(workflow.completion_requirements)) is None


def test_runtime_accepts_custom_workflow() -> None:
    workflow = ResearchWorkflow()
    runtime = AgentRuntime((workflow,))
    assert runtime.workflow_for("any objective") is workflow
