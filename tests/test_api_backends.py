import json
from unittest.mock import MagicMock, patch

from molsim_agent.agent.messages import Message
from molsim_agent.llm.anthropic import AnthropicBackend
from molsim_agent.llm.openai_compatible import GroqBackend, OpenAICompatibleBackend


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


def test_groq_backend_uses_groq_environment_key_without_logging_it(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "secret-groq-key")
    payload = {"choices": [{"message": {"content": "ok"}}]}
    with patch("molsim_agent.llm.openai_compatible.urlopen", return_value=_response(payload)) as open_url:
        result = GroqBackend("llama-3.1-8b-instant").chat([Message("user", "hello")], [])

    request = open_url.call_args.args[0]
    assert request.full_url == "https://api.groq.com/openai/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer secret-groq-key"
    assert "tools" not in json.loads(request.data)
    assert result.content == "ok"
