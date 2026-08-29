import pytest
from backend.behavior import (
    AnswerBehavior,
)
from backend.generation.claim_grounding_validator import (
    GeneratedClaimGroundingValidator,
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
from backend.routing.answerability import (
    AnswerabilityResult,
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

class RecordingAnswerabilityProvider:
    def __init__(
        self,
        score: float,
    ) -> None:
        self._score = score
        self.call_count = 0

    def score(
        self,
        question: str,
        evidence_texts: tuple[str, ...],
    ) -> AnswerabilityResult:
        self.call_count += 1

        return AnswerabilityResult(
            scores=tuple(
                self._score
                for _ in evidence_texts
            )
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

def _validator(
    score: float = 0.95,
) -> tuple[
    GeneratedClaimGroundingValidator,
    RecordingAnswerabilityProvider,
]:
    provider = RecordingAnswerabilityProvider(
        score
    )

    validator = (
        GeneratedClaimGroundingValidator(
            answerability_provider=provider,
            support_threshold=0.80,
        )
    )

    return validator, provider

def test_valid_generation_is_released():
    generator = FakeGroundedAnswerGenerator(
        "Approval is required [E1]."
    )

    validator, _ = _validator()

    result = (
        generate_validated_grounded_answer(
            _request(),
            generator,
            validator,
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
        "Approval is required [E1]. "
        "The approval must be recorded [E2]."
    )

    validator, _ = _validator()

    result = (
        generate_validated_grounded_answer(
            _request(),
            generator,
            validator,
        )
    )

    assert (
        result.cited_evidence_indexes
        == (1, 2)
    )

def test_missing_citations_are_rejected_before_grounding():
    generator = FakeGroundedAnswerGenerator(
        "Approval is required."
    )

    validator, provider = _validator()

    with pytest.raises(
        GroundedGenerationRejectedError,
        match=(
            "does not cite any approved evidence"
        ),
    ):
        generate_validated_grounded_answer(
            _request(),
            generator,
            validator,
        )

    assert provider.call_count == 0

def test_unknown_citation_is_rejected_before_grounding():
    generator = FakeGroundedAnswerGenerator(
        "Approval is required [E3]."
    )

    validator, provider = _validator()

    with pytest.raises(
        GroundedGenerationRejectedError,
        match=(
            "cites evidence that was not supplied"
        ),
    ):
        generate_validated_grounded_answer(
            _request(),
            generator,
            validator,
        )

    assert provider.call_count == 0

def test_unsupported_claim_is_rejected():
    generator = FakeGroundedAnswerGenerator(
        "Approval must occur within "
        "14 days [E1]."
    )

    validator, provider = _validator(
        score=0.20
    )

    with pytest.raises(
        GroundedGenerationRejectedError,
        match=(
            "is not sufficiently supported"
        ),
    ):
        generate_validated_grounded_answer(
            _request(),
            generator,
            validator,
        )

    assert provider.call_count == 1

def test_rejected_generated_text_is_not_exposed():
    unsupported_text = (
        "Invented policy requirement [E9]."
    )

    generator = FakeGroundedAnswerGenerator(
        unsupported_text
    )

    validator, _ = _validator()

    with pytest.raises(
        GroundedGenerationRejectedError
    ) as error:
        generate_validated_grounded_answer(
            _request(),
            generator,
            validator,
        )

    assert (
        unsupported_text
        not in str(error.value)
    )

def test_generator_is_called_once():
    generator = FakeGroundedAnswerGenerator(
        "Approval is required [E1]."
    )

    validator, _ = _validator()

    generate_validated_grounded_answer(
        _request(),
        generator,
        validator,
    )

    assert generator.call_count == 1