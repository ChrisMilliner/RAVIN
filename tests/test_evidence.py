import pytest
from backend.core.config import DEFAULT_EVIDENCE_THRESHOLD
from backend.core.evidence import assess_evidence
from backend.core.fixtures import POLICY_FIXTURES
from backend.core.models import (
    EvidenceSufficiency,
    RetrievedEvidence,
)
from backend.core.retrieval import retrieve_evidence

def test_strong_extension_evidence_is_sufficient():
    evidence = retrieve_evidence(
        "When can a student request an assessment extension?",
        POLICY_FIXTURES,
    )

    assessment = assess_evidence(evidence)

    assert assessment.sufficiency is EvidenceSufficiency.SUFFICIENT
    assert assessment.best_score >= DEFAULT_EVIDENCE_THRESHOLD
    assert len(assessment.supporting_evidence) >= 1
    assert assessment.supporting_evidence[0].policy_id == "POL-EXT-001"

def test_weak_unrelated_evidence_is_insufficient():
    evidence = retrieve_evidence(
        "Which colour should I paint my car?",
        POLICY_FIXTURES,
    )

    assessment = assess_evidence(evidence)

    assert assessment.sufficiency is EvidenceSufficiency.INSUFFICIENT
    assert assessment.best_score < DEFAULT_EVIDENCE_THRESHOLD
    assert assessment.supporting_evidence == ()

def test_no_evidence_is_insufficient():
    assessment = assess_evidence(())

    assert assessment.sufficiency is EvidenceSufficiency.INSUFFICIENT
    assert assessment.best_score == 0.0
    assert assessment.supporting_evidence == ()

def test_evidence_at_threshold_is_sufficient():
    evidence = (
        RetrievedEvidence(
            policy_id="POL-TEST-001",
            policy_title="Threshold Test Policy",
            source_url="https://example.invalid/policies/threshold-test",
            text="Representative evidence.",
            relevance_score=0.5,
        ),
    )

    assessment = assess_evidence(evidence)

    assert assessment.sufficiency is EvidenceSufficiency.SUFFICIENT
    assert assessment.best_score == 0.5
    assert assessment.supporting_evidence == evidence

def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError):
        assess_evidence((), threshold=1.5)

def test_custom_threshold_changes_sufficiency_decision():
    evidence = (
        RetrievedEvidence(
            policy_id="POL-TEST-001",
            policy_title="Threshold Test Policy",
            source_url="https://example.invalid/policies/threshold-test",
            text="Representative evidence.",
            relevance_score=0.6,
        ),
    )

    lower_threshold = assess_evidence(
        evidence,
        threshold=0.5,
    )

    higher_threshold = assess_evidence(
        evidence,
        threshold=0.7,
    )

    assert lower_threshold.sufficiency is EvidenceSufficiency.SUFFICIENT
    assert higher_threshold.sufficiency is EvidenceSufficiency.INSUFFICIENT