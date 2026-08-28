import pytest
from backend.behavior import AnswerBehavior
from backend.generation.grounded_generator import (
    GroundedAnswerGenerator,
    GroundedGenerationRequest,
    GroundedGenerationResult,
    generate_grounded_answer,
)

class FakeGroundedAnswerGenerator:
    def generate(
        self,
        request: GroundedGenerationRequest,
    ) -> GroundedGenerationResult:
        return GroundedGenerationResult(
            text="Grounded response."
        )

class InvalidGroundedAnswerGenerator:
    def generate(
        self,
        request: GroundedGenerationRequest,
    ):
        return "Invalid result."

def _request(
    behavior: AnswerBehavior = (
        AnswerBehavior.DIRECT_ANSWER
    ),
) -> GroundedGenerationRequest:
    return GroundedGenerationRequest(
        question=(
            "What does the policy require?"
        ),
        behavior=behavior,
        evidence_texts=(
            "Relevant policy evidence.",
        ),
    )

def test_direct_answer_request_is_valid():
    request = _request(
        AnswerBehavior.DIRECT_ANSWER
    )

    assert (
        request.behavior
        == AnswerBehavior.DIRECT_ANSWER
    )

def test_grounded_overview_request_is_valid():
    request = _request(
        AnswerBehavior.GROUNDED_OVERVIEW
    )

    assert (
        request.behavior
        == AnswerBehavior.GROUNDED_OVERVIEW
    )

@pytest.mark.parametrize(
    "behavior",
    (
        AnswerBehavior.CLARIFY,
        AnswerBehavior.NO_GROUNDED_ANSWER,
    ),
)

def test_non_generation_behavior_is_rejected(
    behavior,
):
    with pytest.raises(
        ValueError,
        match=(
            "Grounded generation is only allowed"
        ),
    ):
        _request(
            behavior
        )

def test_empty_evidence_is_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "Grounded generation requires evidence"
        ),
    ):
        GroundedGenerationRequest(
            question="Question?",
            behavior=(
                AnswerBehavior.DIRECT_ANSWER
            ),
            evidence_texts=(),
        )

def test_empty_evidence_text_is_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "Generation evidence cannot contain"
        ),
    ):
        GroundedGenerationRequest(
            question="Question?",
            behavior=(
                AnswerBehavior.DIRECT_ANSWER
            ),
            evidence_texts=(
                "Evidence.",
                " ",
            ),
        )

def test_generator_result_is_returned():
    result = generate_grounded_answer(
        _request(),
        FakeGroundedAnswerGenerator(),
    )

    assert (
        result.text
        == "Grounded response."
    )

def test_invalid_generator_result_is_rejected():
    generator = InvalidGroundedAnswerGenerator()

    with pytest.raises(
        ValueError,
        match=(
            "Grounded answer generator must return"
        ),
    ):
        generate_grounded_answer(
            _request(),
            generator,  # type: ignore[arg-type]
        )