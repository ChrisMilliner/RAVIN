import pytest
from backend.behavior import AnswerBehavior
from backend.generation.grounded_generator import (
    GroundedGenerationRequest,
)
from backend.generation.llm_grounded_generator import (
    LlmGroundedAnswerGenerator,
)

class RecordingLanguageModelProvider:
    def __init__(
        self,
        response: str = (
            "The policy requires approval [E1]."
        ),
    ) -> None:
        self.response = response
        self.system_prompt = None
        self.user_prompt = None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        return self.response

def _request(
    behavior: AnswerBehavior = (
        AnswerBehavior.DIRECT_ANSWER
    ),
    evidence_texts: tuple[str, ...] = (
        "Approval is required.",
    ),
) -> GroundedGenerationRequest:
    return GroundedGenerationRequest(
        question=(
            "What does the policy require?"
        ),
        behavior=behavior,
        evidence_texts=evidence_texts,
    )

def test_direct_answer_calls_language_model():
    provider = (
        RecordingLanguageModelProvider()
    )

    generator = LlmGroundedAnswerGenerator(
        provider
    )

    result = generator.generate(
        _request()
    )

    assert result.text == (
        "The policy requires approval [E1]."
    )

    assert provider.system_prompt is not None
    assert provider.user_prompt is not None

def test_direct_answer_prompt_is_constrained():
    provider = (
        RecordingLanguageModelProvider()
    )

    generator = LlmGroundedAnswerGenerator(
        provider
    )

    generator.generate(
        _request()
    )

    system_prompt = provider.system_prompt

    assert system_prompt is not None

    assert (
        "Use only the approved evidence"
        in system_prompt
    )

    assert (
        "Do not use outside knowledge"
        in system_prompt
    )

    assert (
        "Answer the focused question directly"
        in system_prompt
    )

def test_overview_uses_overview_instruction():
    provider = (
        RecordingLanguageModelProvider()
    )

    generator = LlmGroundedAnswerGenerator(
        provider
    )

    generator.generate(
        _request(
            behavior=(
                AnswerBehavior.GROUNDED_OVERVIEW
            )
        )
    )

    system_prompt = provider.system_prompt

    assert system_prompt is not None

    assert (
        "Provide a concise grounded overview"
        in system_prompt
    )

def test_evidence_is_numbered_in_user_prompt():
    provider = (
        RecordingLanguageModelProvider()
    )

    generator = LlmGroundedAnswerGenerator(
        provider
    )

    generator.generate(
        _request(
            evidence_texts=(
                "First evidence.",
                "Second evidence.",
            )
        )
    )

    user_prompt = provider.user_prompt

    assert user_prompt is not None

    assert (
        "[E1]\nFirst evidence."
        in user_prompt
    )

    assert (
        "[E2]\nSecond evidence."
        in user_prompt
    )

def test_question_is_in_user_prompt():
    provider = (
        RecordingLanguageModelProvider()
    )

    generator = LlmGroundedAnswerGenerator(
        provider
    )

    generator.generate(
        _request()
    )

    user_prompt = provider.user_prompt

    assert user_prompt is not None

    assert (
        "What does the policy require?"
        in user_prompt
    )

def test_language_model_output_is_trimmed():
    provider = (
        RecordingLanguageModelProvider(
            response=(
                "  Grounded answer [E1].  "
            )
        )
    )

    generator = LlmGroundedAnswerGenerator(
        provider
    )

    result = generator.generate(
        _request()
    )

    assert (
        result.text
        == "Grounded answer [E1]."
    )

def test_empty_language_model_output_is_rejected():
    provider = (
        RecordingLanguageModelProvider(
            response=" "
        )
    )

    generator = LlmGroundedAnswerGenerator(
        provider
    )

    with pytest.raises(
        ValueError,
        match=(
            "Language model returned an "
            "empty response"
        ),
    ):
        generator.generate(
            _request()
        )