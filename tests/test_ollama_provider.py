import json
import pytest
import backend.llm.ollama_provider as ollama_module
from backend.llm.ollama_provider import (
    OllamaLanguageModelProvider,
)
from backend.llm.ollama_provider import (
    OllamaLanguageModelProvider,
)

class FakeResponse:
    def __init__(
        self,
        payload: dict,
    ) -> None:
        self._payload = payload

    def __enter__(
        self,
    ):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        return None

    def read(
        self,
    ) -> bytes:
        return json.dumps(
            self._payload
        ).encode(
            "utf-8"
        )

def test_provider_returns_generated_text(
    monkeypatch,
):
    def fake_urlopen(
        request,
        timeout,
    ):
        return FakeResponse(
            {
                "message": {
                    "content": (
                        "Grounded answer [E1]."
                    )
                }
            }
        )

    monkeypatch.setattr(
        ollama_module,
        "urlopen",
        fake_urlopen,
    )

    provider = (
        OllamaLanguageModelProvider(
            model_name=(
                "qwen3:4b-instruct"
            )
        )
    )

    result = provider.generate(
        "System prompt.",
        "User prompt.",
    )

    assert (
        result
        == "Grounded answer [E1]."
    )

def test_provider_sends_expected_request(
    monkeypatch,
):
    captured = {}

    def fake_urlopen(
        request,
        timeout,
    ):
        captured["url"] = (
            request.full_url
        )

        captured["timeout"] = timeout

        captured["payload"] = (
            json.loads(
                request.data.decode(
                    "utf-8"
                )
            )
        )

        return FakeResponse(
            {
                "message": {
                    "content": "Answer."
                }
            }
        )

    monkeypatch.setattr(
        ollama_module,
        "urlopen",
        fake_urlopen,
    )

    provider = (
        OllamaLanguageModelProvider(
            model_name=(
                "qwen3:4b-instruct"
            ),
            timeout_seconds=30.0,
        )
    )

    provider.generate(
        "System prompt.",
        "User prompt.",
    )

    assert captured["url"] == (
        "http://localhost:11434/api/chat"
    )

    assert captured["timeout"] == 30.0

    payload = captured["payload"]

    assert payload["model"] == (
        "qwen3:4b-instruct"
    )

    assert payload["stream"] is False

    assert (
        payload["options"]["temperature"]
        == 0
    )

    assert payload["messages"] == [
        {
            "role": "system",
            "content": "System prompt.",
        },
        {
            "role": "user",
            "content": "User prompt.",
        },
    ]

def test_custom_base_url_is_supported(
    monkeypatch,
):
    captured = {}

    def fake_urlopen(
        request,
        timeout,
    ):
        captured["url"] = (
            request.full_url
        )

        return FakeResponse(
            {
                "message": {
                    "content": "Answer."
                }
            }
        )

    monkeypatch.setattr(
        ollama_module,
        "urlopen",
        fake_urlopen,
    )

    provider = (
        OllamaLanguageModelProvider(
            model_name="model",
            base_url=(
                "http://example.test:11434/"
            ),
        )
    )

    provider.generate(
        "System.",
        "User.",
    )

    assert captured["url"] == (
        "http://example.test:11434/api/chat"
    )

def test_empty_model_name_is_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "Ollama model name cannot be empty"
        ),
    ):
        OllamaLanguageModelProvider(
            model_name=" "
        )

def test_empty_base_url_is_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "Ollama base URL cannot be empty"
        ),
    ):
        OllamaLanguageModelProvider(
            model_name="model",
            base_url=" ",
        )

def test_invalid_timeout_is_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "Ollama timeout must be greater "
            "than zero"
        ),
    ):
        OllamaLanguageModelProvider(
            model_name="model",
            timeout_seconds=0,
        )

def test_missing_message_is_rejected(
    monkeypatch,
):
    monkeypatch.setattr(
        ollama_module,
        "urlopen",
        lambda request, timeout: (
            FakeResponse({})
        ),
    )

    provider = (
        OllamaLanguageModelProvider(
            model_name="model"
        )
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "does not contain a message"
        ),
    ):
        provider.generate(
            "System.",
            "User.",
        )

def test_missing_content_is_rejected(
    monkeypatch,
):
    monkeypatch.setattr(
        ollama_module,
        "urlopen",
        lambda request, timeout: (
            FakeResponse(
                {
                    "message": {}
                }
            )
        ),
    )

    provider = (
        OllamaLanguageModelProvider(
            model_name="model"
        )
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "does not contain generated text"
        ),
    ):
        provider.generate(
            "System.",
            "User.",
        )