import pytest
from backend.behavior import (
    AnswerBehavior,
)
from backend.generation.grounded_generator import (
    GroundedGenerationRequest,
    GroundedGenerationResult,
)
from backend.generation.release_gate import (
    GroundedGenerationRejectedError,
    ReleasedGroundedAnswer,
    generate_validated_grounded_answer,
)

class FakeGroundedAnswerGenerator:
    def __init__(
        self,
        text: str,
    ) -> None:
        self._text = text
        self.call_count = 0

    def generate(
        self,
        request: GroundedGenerationRequest,
    ) -> GroundedGenerationResult:
        self.call_count += 1

        return GroundedGenerationResult(
            text=self._text
        )

def _request() -> GroundedGenerationRequest:
    return GroundedGenerationRequest(
        question=(
            "What does the policy require?"
        ),
        behavior=(
            AnswerBehavior.DIRECT_ANSWER
        ),
        evidence_texts=(
            "Approval is required.",
            "The approval must be recorded.",
        ),
    )

def test_valid_generation_is_released():
    generator = FakeGroundedAnswerGenerator(
        "Approval is required [E1]."
    )

    result = (
        generate_validated_grounded_answer(
            _request(),
            generator,
        )
    )

    assert isinstance(
        result,
        ReleasedGroundedAnswer,
    )

    assert (
        result.text
        == "Approval is required [E1]."
    )

    assert (
        result.cited_evidence_indexes
        == (1,)
    )

def test_multiple_valid_citations_are_released():
    generator = FakeGroundedAnswerGenerator(
        "Approval is required [E1], "
        "and it must be recorded [E2]."
    )

    result = (
        generate_validated_grounded_answer(
            _request(),
            generator,
        )
    )

    assert (
        result.cited_evidence_indexes
        == (1, 2)
    )

def test_missing_citations_are_rejected():
    generator = FakeGroundedAnswerGenerator(
        "Approval is required."
    )

    with pytest.raises(
        GroundedGenerationRejectedError,
        match=(
            "does not cite any approved evidence"
        ),
    ):
        generate_validated_grounded_answer(
            _request(),
            generator,
        )

def test_unknown_citation_is_rejected():
    generator = FakeGroundedAnswerGenerator(
        "Approval is required [E3]."
    )

    with pytest.raises(
        GroundedGenerationRejectedError,
        match=(
            "cites evidence that was not supplied"
        ),
    ):
        generate_validated_grounded_answer(
            _request(),
            generator,
        )

def test_rejected_generated_text_is_not_exposed():
    unsupported_text = (
        "Invented policy requirement [E9]."
    )

    generator = FakeGroundedAnswerGenerator(
        unsupported_text
    )

    with pytest.raises(
        GroundedGenerationRejectedError
    ) as error:
        generate_validated_grounded_answer(
            _request(),
            generator,
        )

    assert (
        unsupported_text
        not in str(error.value)
    )

def test_generator_is_called_once():
    generator = FakeGroundedAnswerGenerator(
        "Approval is required [E1]."
    )

    generate_validated_grounded_answer(
        _request(),
        generator,
    )

    assert generator.call_count == 1