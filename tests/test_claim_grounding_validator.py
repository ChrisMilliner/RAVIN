import pytest
from backend.behavior import (
    AnswerBehavior,
)
from backend.generation.claim_grounding_validator import (
    GeneratedClaimGroundingValidator,
)
from backend.generation.entailment import (
    EntailmentPair,
)
from backend.generation.grounded_generator import (
    GroundedGenerationRequest,
    GroundedGenerationResult,
)

class FakeEntailmentProvider:
    def __init__(
        self,
        scores: tuple[float, ...],
    ) -> None:
        self._scores = scores
        self.received_pairs: list[
            tuple[
                EntailmentPair,
                ...
            ]
        ] = []

    def score_entailment(
        self,
        pairs: tuple[
            EntailmentPair,
            ...
        ],
    ) -> tuple[float, ...]:
        self.received_pairs.append(
            pairs
        )

        return self._scores

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
    provider = FakeEntailmentProvider(
        scores=(0.95,)
    )

    validator = (
        GeneratedClaimGroundingValidator(
            entailment_provider=provider,
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
    provider = FakeEntailmentProvider(
        scores=(0.20,)
    )

    validator = (
        GeneratedClaimGroundingValidator(
            entailment_provider=provider,
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
    provider = FakeEntailmentProvider(
        scores=(0.95,)
    )

    validator = (
        GeneratedClaimGroundingValidator(
            entailment_provider=provider,
            support_threshold=0.80,
        )
    )

    validator.validate(
        _request(),
        _result(
            "Approval must be recorded [E2]."
        ),
    )

    pairs = provider.received_pairs[0]

    assert tuple(
        pair.premise
        for pair in pairs
    ) == (
        "Approval must be recorded.",
    )

def test_citation_marker_is_removed_before_scoring():
    provider = FakeEntailmentProvider(
        scores=(0.95,)
    )

    validator = (
        GeneratedClaimGroundingValidator(
            entailment_provider=provider,
            support_threshold=0.80,
        )
    )

    validator.validate(
        _request(),
        _result(
            "Approval must be recorded [E2]."
        ),
    )

    pairs = provider.received_pairs[0]

    assert tuple(
        pair.hypothesis
        for pair in pairs
    ) == (
        "Approval must be recorded.",
    )

def test_uncited_claim_is_invalid_without_scoring():
    provider = FakeEntailmentProvider(
        scores=(0.95,)
    )

    validator = (
        GeneratedClaimGroundingValidator(
            entailment_provider=provider,
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
    assert provider.received_pairs == []

def test_unknown_citation_is_invalid_without_scoring():
    provider = FakeEntailmentProvider(
        scores=(0.95,)
    )

    validator = (
        GeneratedClaimGroundingValidator(
            entailment_provider=provider,
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
    assert provider.received_pairs == []

def test_multiple_supported_claims_are_valid():
    provider = FakeEntailmentProvider(
        scores=(0.95,)
    )

    validator = (
        GeneratedClaimGroundingValidator(
            entailment_provider=provider,
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
    assert len(provider.received_pairs) == 2

def test_three_unit_window_can_support_compound_claim():
    request = GroundedGenerationRequest(
        question="How is a review handled?",
        behavior=(
            AnswerBehavior.DIRECT_ANSWER
        ),
        evidence_texts=(
            (
                "Students may seek a review through "
                "the UAC. "
                "The review must be requested within "
                "20 business days. "
                "The UAC decision is final."
            ),
        ),
    )

    provider = FakeEntailmentProvider(
        scores=(
            0.10,
            0.20,
            0.97,
            0.15,
            0.30,
            0.40,
        )
    )

    validator = (
        GeneratedClaimGroundingValidator(
            entailment_provider=provider,
            support_threshold=0.80,
        )
    )

    result = validator.validate(
        request,
        _result(
            (
                "A review may be sought through "
                "the UAC within 20 business days, "
                "and the UAC decision is final "
                "[E1]."
            )
        ),
    )

    assert result.valid is True

    assert (
        result.claims[0].score
        == 0.97
    )

    pairs = provider.received_pairs[0]

    assert len(pairs) == 6

    assert pairs[2].premise == (
        "Students may seek a review through "
        "the UAC. "
        "The review must be requested within "
        "20 business days. "
        "The UAC decision is final."
    )

def test_strongest_entailment_score_controls_support():
    provider = FakeEntailmentProvider(
        scores=(
            0.10,
            0.91,
            0.30,
        )
    )

    request = GroundedGenerationRequest(
        question="What happens?",
        behavior=(
            AnswerBehavior.DIRECT_ANSWER
        ),
        evidence_texts=(
            "First rule. Second rule.",
        ),
    )

    validator = (
        GeneratedClaimGroundingValidator(
            entailment_provider=provider,
            support_threshold=0.80,
        )
    )

    result = validator.validate(
        request,
        _result(
            "The supported rule applies [E1]."
        ),
    )

    assert result.valid is True

    assert (
        result.claims[0].score
        == 0.91
    )

def test_invalid_threshold_is_rejected():
    provider = FakeEntailmentProvider(
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
            entailment_provider=provider,
            support_threshold=1.1,
        )