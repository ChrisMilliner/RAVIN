from backend.behavior import (
    AnswerBehavior,
)
from backend.generation.citation_validator import (
    validate_generation_citations,
)
from backend.generation.grounded_generator import (
    GroundedGenerationRequest,
    GroundedGenerationResult,
)

def _request(
    evidence_texts: tuple[str, ...] = (
        "First approved evidence.",
        "Second approved evidence.",
    ),
) -> GroundedGenerationRequest:
    return GroundedGenerationRequest(
        question=(
            "What does the policy require?"
        ),
        behavior=(
            AnswerBehavior.DIRECT_ANSWER
        ),
        evidence_texts=evidence_texts,
    )

def _result(
    text: str,
) -> GroundedGenerationResult:
    return GroundedGenerationResult(
        text=text
    )

def test_valid_single_citation_is_accepted():
    validation = (
        validate_generation_citations(
            _request(),
            _result(
                "Approval is required [E1]."
            ),
        )
    )

    assert validation.valid is True
    assert (
        validation.cited_evidence_indexes
        == (1,)
    )

def test_valid_multiple_citations_are_accepted():
    validation = (
        validate_generation_citations(
            _request(),
            _result(
                "Approval is required [E1]. "
                "Another condition applies [E2]."
            ),
        )
    )

    assert validation.valid is True
    assert (
        validation.cited_evidence_indexes
        == (1, 2)
    )

def test_duplicate_citations_are_deduplicated():
    validation = (
        validate_generation_citations(
            _request(),
            _result(
                "Approval is required [E1]. "
                "This is confirmed again [E1]."
            ),
        )
    )

    assert validation.valid is True
    assert (
        validation.cited_evidence_indexes
        == (1,)
    )

def test_missing_citation_is_rejected():
    validation = (
        validate_generation_citations(
            _request(),
            _result(
                "Approval is required."
            ),
        )
    )

    assert validation.valid is False
    assert (
        validation.cited_evidence_indexes
        == ()
    )

def test_unknown_citation_is_rejected():
    validation = (
        validate_generation_citations(
            _request(),
            _result(
                "Approval is required [E3]."
            ),
        )
    )

    assert validation.valid is False
    assert (
        validation.cited_evidence_indexes
        == (3,)
    )

def test_mixed_valid_and_unknown_citations_are_rejected():
    validation = (
        validate_generation_citations(
            _request(),
            _result(
                "Approval is required [E1], "
                "and another rule applies [E4]."
            ),
        )
    )

    assert validation.valid is False
    assert (
        validation.cited_evidence_indexes
        == (1, 4)
    )

def test_highest_available_citation_is_valid():
    validation = (
        validate_generation_citations(
            _request(),
            _result(
                "The second evidence applies [E2]."
            ),
        )
    )

    assert validation.valid is True
    assert (
        validation.cited_evidence_indexes
        == (2,)
    )