import pytest
from backend.core.fixtures import POLICY_FIXTURES
from backend.core.models import ResponseOutcome
from backend.core.messages import INSUFFICIENT_EVIDENCE_MESSAGE
from backend.core.response import build_grounded_response

def test_supported_extension_question_returns_grounded_response():
    response = build_grounded_response(
        "When can a student request an assessment extension?",
        POLICY_FIXTURES,
    )

    assert response.outcome is ResponseOutcome.SUPPORTED
    assert response.answer is not None
    assert len(response.sources) >= 1
    assert response.sources[0].policy_id == "POL-EXT-001"

def test_supported_response_preserves_source_provenance():
    response = build_grounded_response(
        "When can a student request an assessment extension?",
        POLICY_FIXTURES,
    )

    source = response.sources[0]

    assert source.policy_id == "POL-EXT-001"
    assert source.policy_title == "Example Assessment Extension Policy"
    assert (
        source.source_url
        == "https://example.invalid/policies/assessment-extension"
    )

def test_unrelated_question_returns_controlled_insufficient_evidence():
    response = build_grounded_response(
        "Which colour should I paint my car?",
        POLICY_FIXTURES,
    )

    assert response.outcome is ResponseOutcome.INSUFFICIENT_EVIDENCE
    assert response.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert response.sources == ()

def test_empty_question_is_rejected_before_retrieval():
    with pytest.raises(
        ValueError,
        match="Question must contain non-whitespace text.",
    ):
        build_grounded_response(
            "   ",
            POLICY_FIXTURES,
        )

def test_custom_threshold_can_force_controlled_refusal():
    response = build_grounded_response(
        "When can a student request an assessment extension?",
        POLICY_FIXTURES,
        threshold=1.0,
    )

    assert response.outcome is ResponseOutcome.INSUFFICIENT_EVIDENCE
    assert response.sources == ()