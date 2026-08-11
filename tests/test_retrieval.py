from backend.core.fixtures import POLICY_FIXTURES
from backend.core.retrieval import retrieve_evidence

def test_extension_question_retrieves_extension_policy():
    evidence = retrieve_evidence(
        "When can a student request an assessment extension?",
        POLICY_FIXTURES,
    )

    assert len(evidence) >= 1
    assert evidence[0].policy_id == "POL-EXT-001"
    assert evidence[0].policy_title == "Example Assessment Extension Policy"

def test_academic_integrity_question_retrieves_integrity_policy():
    evidence = retrieve_evidence(
        "What does the academic integrity policy say about plagiarism?",
        POLICY_FIXTURES,
    )

    assert len(evidence) >= 1
    assert evidence[0].policy_id == "POL-AI-001"

def test_unrelated_question_returns_only_weak_or_no_evidence():
    evidence = retrieve_evidence(
        "Which colour should I paint my car?",
        POLICY_FIXTURES,
    )

    assert all(
        item.relevance_score < 0.5
        for item in evidence
    )

def test_retrieved_evidence_preserves_source_provenance():
    evidence = retrieve_evidence(
        "Can a student request an assessment extension?",
        POLICY_FIXTURES,
    )

    first_match = evidence[0]

    assert first_match.policy_id == "POL-EXT-001"
    assert first_match.policy_title == "Example Assessment Extension Policy"
    assert (
        first_match.source_url
        == "https://example.invalid/policies/assessment-extension"
    )

def test_retrieval_orders_stronger_matches_first():
    evidence = retrieve_evidence(
        "student assessment extension request deadline",
        POLICY_FIXTURES,
    )

    assert len(evidence) >= 1
    assert evidence[0].policy_id == "POL-EXT-001"
    assert evidence[0].relevance_score > 0