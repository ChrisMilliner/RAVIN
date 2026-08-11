import pytest
from dataclasses import FrozenInstanceError
from backend.core.models import (
    GroundedResponse,
    PolicyDocument,
    ResponseOutcome,
    RetrievedEvidence,
)

def test_policy_document_preserves_source_metadata():
    policy = PolicyDocument(
        policy_id="POL-001",
        title="Example Assessment Extension Policy",
        source_url="https://example.invalid/policies/extensions",
        status="current",
        text="Students may request an extension under defined conditions.",
    )

    assert policy.policy_id == "POL-001"
    assert policy.title == "Example Assessment Extension Policy"
    assert policy.source_url == "https://example.invalid/policies/extensions"
    assert policy.status == "current"

def test_policy_document_is_immutable():
    policy = PolicyDocument(
        policy_id="POL-001",
        title="Example Assessment Extension Policy",
        source_url="https://example.invalid/policies/extensions",
        status="current",
        text="Representative policy text.",
    )

    with pytest.raises(FrozenInstanceError):
        setattr(policy, "title", "Changed title")


def test_supported_response_can_retain_evidence():
    evidence = RetrievedEvidence(
        policy_id="POL-001",
        policy_title="Example Assessment Extension Policy",
        source_url="https://example.invalid/policies/extensions",
        text="Students may request an extension under defined conditions.",
        relevance_score=1.0,
    )

    response = GroundedResponse(
        outcome=ResponseOutcome.SUPPORTED,
        answer="A student may request an extension under the policy conditions.",
        sources=(evidence,),
    )

    assert response.outcome is ResponseOutcome.SUPPORTED
    assert response.answer is not None
    assert response.sources[0].policy_id == "POL-001"

def test_insufficient_evidence_response_contains_no_sources():
    response = GroundedResponse(
        outcome=ResponseOutcome.INSUFFICIENT_EVIDENCE,
        answer="Insufficient policy evidence was found.",
        sources=(),
    )

    assert response.outcome is ResponseOutcome.INSUFFICIENT_EVIDENCE
    assert response.answer == "Insufficient policy evidence was found."
    assert response.sources == ()