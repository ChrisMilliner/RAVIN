from backend.evaluation.matching import (
    matches_expected_evidence,
)
from backend.evaluation.models import ExpectedEvidence
from backend.ingestion.models import PolicyChunk

def make_chunk(
    policy_id: str = "208",
    heading_path: tuple[str, ...] = (
        "Section 4 - Key Decisions",
    ),
) -> PolicyChunk:
    return PolicyChunk(
        policy_id=policy_id,
        policy_title="Academic Dress Policy",
        source_url=(
            "https://policies.latrobe.edu.au/"
            f"document/view.php?id={policy_id}"
        ),
        status="Current",
        effective_date=None,
        review_date=None,
        chunk_index=0,
        text="Example policy evidence.",
        heading_path=heading_path,
    )

def test_exact_policy_and_heading_path_matches():
    chunk = make_chunk()

    expected = ExpectedEvidence(
        policy_id="208",
        heading_path=(
            "Section 4 - Key Decisions",
        ),
    )

    assert matches_expected_evidence(
        chunk,
        expected,
    )

def test_different_policy_does_not_match():
    chunk = make_chunk(
        policy_id="208",
    )

    expected = ExpectedEvidence(
        policy_id="169",
        heading_path=(
            "Section 4 - Key Decisions",
        ),
    )

    assert not matches_expected_evidence(
        chunk,
        expected,
    )

def test_different_heading_path_does_not_match():
    chunk = make_chunk(
        heading_path=(
            "Section 5 - Policy Statement",
        ),
    )

    expected = ExpectedEvidence(
        policy_id="208",
        heading_path=(
            "Section 4 - Key Decisions",
        ),
    )

    assert not matches_expected_evidence(
        chunk,
        expected,
    )

def test_parent_expected_path_does_not_match_by_default():
    chunk = make_chunk(
        heading_path=(
            "Section 6 - Procedures",
            "Part A - Entry Criteria",
            "Single Subjects",
        ),
    )

    expected = ExpectedEvidence(
        policy_id="208",
        heading_path=(
            "Section 6 - Procedures",
            "Part A - Entry Criteria",
        ),
    )

    assert not matches_expected_evidence(
        chunk,
        expected,
    )

def test_parent_expected_path_matches_when_descendants_allowed():
    chunk = make_chunk(
        heading_path=(
            "Section 6 - Procedures",
            "Part A - Entry Criteria",
            "Single Subjects",
        ),
    )

    expected = ExpectedEvidence(
        policy_id="208",
        heading_path=(
            "Section 6 - Procedures",
            "Part A - Entry Criteria",
        ),
        allow_descendants=True,
    )

    assert matches_expected_evidence(
        chunk,
        expected,
    )

def test_expected_path_deeper_than_chunk_does_not_match():
    chunk = make_chunk(
        heading_path=(
            "Section 6 - Procedures",
        ),
    )

    expected = ExpectedEvidence(
        policy_id="208",
        heading_path=(
            "Section 6 - Procedures",
            "Part A - Entry Criteria",
        ),
        allow_descendants=True,
    )

    assert not matches_expected_evidence(
        chunk,
        expected,
    )

def test_required_text_fragment_matches_chunk_text():
    original = make_chunk()

    chunk = PolicyChunk(
        policy_id=original.policy_id,
        policy_title=original.policy_title,
        source_url=original.source_url,
        status=original.status,
        effective_date=original.effective_date,
        review_date=original.review_date,
        chunk_index=original.chunk_index,
        text=(
            "The student receives no intervention "
            "or communication."
        ),
        heading_path=original.heading_path,
    )

    expected = ExpectedEvidence(
        policy_id="208",
        heading_path=(
            "Section 4 - Key Decisions",
        ),
        text_contains=(
            "NO   INTERVENTION or communication"
        ),
    )

    assert matches_expected_evidence(
        chunk,
        expected,
    )

def test_missing_required_text_fragment_does_not_match():
    chunk = make_chunk()

    expected = ExpectedEvidence(
        policy_id="208",
        heading_path=(
            "Section 4 - Key Decisions",
        ),
        text_contains=(
            "This text is not present."
        ),
    )

    assert not matches_expected_evidence(
        chunk,
        expected,
    )

def test_text_fragment_does_not_override_wrong_heading():
    original = make_chunk(
        heading_path=(
            "Section 5 - Policy Statement",
        ),
    )

    chunk = PolicyChunk(
        policy_id=original.policy_id,
        policy_title=original.policy_title,
        source_url=original.source_url,
        status=original.status,
        effective_date=original.effective_date,
        review_date=original.review_date,
        chunk_index=original.chunk_index,
        text="Example policy evidence.",
        heading_path=original.heading_path,
    )

    expected = ExpectedEvidence(
        policy_id="208",
        heading_path=(
            "Section 4 - Key Decisions",
        ),
        text_contains="Example policy evidence",
    )

    assert not matches_expected_evidence(
        chunk,
        expected,
    )