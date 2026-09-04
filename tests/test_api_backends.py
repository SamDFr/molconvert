import json
from unittest.mock import MagicMock, patch

from molsim_agent.agent.messages import Message
from molsim_agent.llm.anthropic import AnthropicBackend
from molsim_agent.llm.openai_compatible import OpenAICompatibleBackend


def _response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode()
    response.__enter__.return_value = response
    return response


def test_openai_compatible_normalizes_tool_call() -> None:
    payload = {"choices": [{"message": {"content": "checking", "tool_calls": [{"id": "1", "function": {"name": "list_directory", "arguments": '{"path":"."}'}}]}}]}
    with patch("molsim_agent.llm.openai_compatible.urlopen", return_value=_response(payload)):
        result = OpenAICompatibleBackend("test", api_key="key").chat([Message("user", "list")], [])
    assert result.content == "checking"
    assert result.tool_calls[0].arguments == {"path": "."}


def test_anthropic_normalizes_tool_use() -> None:
    payload = {"content": [{"type": "text", "text": "checking"}, {"type": "tool_use", "id": "1", "name": "list_directory", "input": {"path": "."}}]}
    with patch("molsim_agent.llm.anthropic.urlopen", return_value=_response(payload)):
        result = AnthropicBackend("claude-test", api_key="key").chat([Message("user", "list")], [])
    assert result.content == "checking"
    assert result.tool_calls[0].name == "list_directory"
