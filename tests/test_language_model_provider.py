import pytest
from backend.llm.provider import (
    generate_text,
)

class FakeLanguageModelProvider:
    def __init__(
        self,
        response: str,
    ) -> None:
        self.response = response
        self.system_prompt: str | None = None
        self.user_prompt: str | None = None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        return self.response

def test_generate_text_passes_prompts_to_provider():
    provider = FakeLanguageModelProvider(
        "focused"
    )

    response = generate_text(
        provider,
        system_prompt=(
            "Classify the question."
        ),
        user_prompt=(
            "Who approves academic dress?"
        ),
    )

    assert response == "focused"

    assert provider.system_prompt == (
        "Classify the question."
    )

    assert provider.user_prompt == (
        "Who approves academic dress?"
    )

def test_generate_text_strips_prompts_and_response():
    provider = FakeLanguageModelProvider(
        "  broad  "
    )

    response = generate_text(
        provider,
        system_prompt=(
            "  Classify the question.  "
        ),
        user_prompt=(
            "  What happens during review?  "
        ),
    )

    assert response == "broad"

    assert provider.system_prompt == (
        "Classify the question."
    )

    assert provider.user_prompt == (
        "What happens during review?"
    )

def test_generate_text_rejects_empty_system_prompt():
    provider = FakeLanguageModelProvider(
        "focused"
    )

    with pytest.raises(
        ValueError,
        match="System prompt cannot be empty.",
    ):
        generate_text(
            provider,
            system_prompt="   ",
            user_prompt="Question?",
        )

def test_generate_text_rejects_empty_user_prompt():
    provider = FakeLanguageModelProvider(
        "focused"
    )

    with pytest.raises(
        ValueError,
        match="User prompt cannot be empty.",
    ):
        generate_text(
            provider,
            system_prompt="Classify.",
            user_prompt="   ",
        )

def test_generate_text_rejects_empty_response():
    provider = FakeLanguageModelProvider(
        "   "
    )

    with pytest.raises(
        ValueError,
        match=(
            "Language model returned an "
            "empty response."
        ),
    ):
        generate_text(
            provider,
            system_prompt="Classify.",
            user_prompt="Question?",
        )