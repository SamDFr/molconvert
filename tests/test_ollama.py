import json
from unittest.mock import MagicMock, patch

from molsim_agent.agent.messages import Message
import pytest

from molsim_agent.llm.ollama import OllamaBackend, OllamaError


def test_ollama_backend_normalizes_tool_calls() -> None:
    response = MagicMock()
    response.read.return_value = json.dumps(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "list_directory", "arguments": {"path": "."}}}
                ],
            }
        }
    ).encode()
    response.__enter__.return_value = response

    with patch("molsim_agent.llm.ollama.urlopen", return_value=response) as mocked:
        result = OllamaBackend("test-model").chat([Message("user", "List files")], [])

    assert result.tool_calls[0].name == "list_directory"
    assert result.tool_calls[0].arguments == {"path": "."}
    sent = json.loads(mocked.call_args.args[0].data)
    assert sent["model"] == "test-model"
    assert sent["stream"] is False


def test_ollama_backend_reports_timeout_cleanly() -> None:
    with patch("molsim_agent.llm.ollama.urlopen", side_effect=TimeoutError):
        with pytest.raises(OllamaError, match="120 seconds"):
            OllamaBackend("test-model").chat([Message("user", "Hello")], [])


def test_ollama_backend_can_disable_thinking() -> None:
    response = MagicMock()
    response.read.return_value = b'{"message":{"content":"done"}}'
    response.__enter__.return_value = response

    with patch("molsim_agent.llm.ollama.urlopen", return_value=response) as mocked:
        OllamaBackend("test-model", think=False).chat([Message("user", "Hello")], [])

    sent = json.loads(mocked.call_args.args[0].data)
    assert sent["think"] is False
