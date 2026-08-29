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
from backend.routing.answerability import (
    AnswerabilityResult,
)

class FakeAnswerabilityProvider:
    def __init__(
        self,
        scores: tuple[float, ...],
    ) -> None:
        self._scores = scores
        self.questions = []
        self.evidence_sets = []

    def score(
        self,
        question: str,
        evidence_texts: tuple[str, ...],
    ) -> AnswerabilityResult:
        self.questions.append(
            question
        )

        self.evidence_sets.append(
            evidence_texts
        )

        return AnswerabilityResult(
            scores=self._scores
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
            "The Dean must approve applications.",
            "Approval must be recorded.",
        ),
    )

def _result(
    text: str,
) -> GroundedGenerationResult:
    return GroundedGenerationResult(
        text=text
    )

def test_supported_claim_is_valid():
    provider = FakeAnswerabilityProvider(
        scores=(0.95,)
    )

    validator = (
        GeneratedClaimGroundingValidator(
            answerability_provider=provider,
            support_threshold=0.80,
        )
    )

    result = validator.validate(
        _request(),
        _result(
            "The Dean must approve "
            "applications [E1]."
        ),
    )

    assert result.valid is True
    assert result.claims[0].supported is True
    assert result.claims[0].score == 0.95

def test_unsupported_claim_is_invalid():
    provider = FakeAnswerabilityProvider(
        scores=(0.20,)
    )

    validator = (
        GeneratedClaimGroundingValidator(
            answerability_provider=provider,
            support_threshold=0.80,
        )
    )

    result = validator.validate(
        _request(),
        _result(
            "Applications must be approved "
            "within 14 days [E1]."
        ),
    )

    assert result.valid is False
    assert result.claims[0].supported is False

def test_only_cited_evidence_is_scored():
    provider = FakeAnswerabilityProvider(
        scores=(0.95,)
    )

    validator = (
        GeneratedClaimGroundingValidator(
            answerability_provider=provider,
            support_threshold=0.80,
        )
    )

    validator.validate(
        _request(),
        _result(
            "Approval must be recorded [E2]."
        ),
    )

    assert provider.evidence_sets == [
        (
            "Approval must be recorded.",
        )
    ]

def test_citation_marker_is_removed_before_scoring():
    provider = FakeAnswerabilityProvider(
        scores=(0.95,)
    )

    validator = (
        GeneratedClaimGroundingValidator(
            answerability_provider=provider,
            support_threshold=0.80,
        )
    )

    validator.validate(
        _request(),
        _result(
            "Approval must be recorded [E2]."
        ),
    )

    assert provider.questions == [
        "Approval must be recorded."
    ]

def test_uncited_claim_is_invalid_without_scoring():
    provider = FakeAnswerabilityProvider(
        scores=(0.95,)
    )

    validator = (
        GeneratedClaimGroundingValidator(
            answerability_provider=provider,
            support_threshold=0.80,
        )
    )

    result = validator.validate(
        _request(),
        _result(
            "Approval must be recorded."
        ),
    )

    assert result.valid is False
    assert provider.questions == []

def test_unknown_citation_is_invalid_without_scoring():
    provider = FakeAnswerabilityProvider(
        scores=(0.95,)
    )

    validator = (
        GeneratedClaimGroundingValidator(
            answerability_provider=provider,
            support_threshold=0.80,
        )
    )

    result = validator.validate(
        _request(),
        _result(
            "Approval must be recorded [E3]."
        ),
    )

    assert result.valid is False
    assert provider.questions == []

def test_multiple_supported_claims_are_valid():
    provider = FakeAnswerabilityProvider(
        scores=(0.95,)
    )

    validator = (
        GeneratedClaimGroundingValidator(
            answerability_provider=provider,
            support_threshold=0.80,
        )
    )

    result = validator.validate(
        _request(),
        _result(
            "The Dean must approve "
            "applications [E1]. "
            "Approval must be recorded [E2]."
        ),
    )

    assert result.valid is True
    assert len(result.claims) == 2

def test_invalid_threshold_is_rejected():
    provider = FakeAnswerabilityProvider(
        scores=(0.95,)
    )

    with pytest.raises(
        ValueError,
        match=(
            "Claim grounding threshold must "
            "be between 0 and 1"
        ),
    ):
        GeneratedClaimGroundingValidator(
            answerability_provider=provider,
            support_threshold=1.1,
        )