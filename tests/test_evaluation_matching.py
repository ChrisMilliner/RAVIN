from backend.evaluation.matching import (
    is_expected_evidence_group_covered,
    matches_expected_evidence,
)
from backend.evaluation.models import (
    ExpectedEvidence,
    ExpectedEvidenceGroup,
)
from backend.ingestion.models import PolicyChunk

def make_chunk(
    policy_id: str = "208",
    heading_path: tuple[str, ...] = (
        "Section 4 - Key Decisions",
    ),
    text: str = "Example policy evidence.",
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
        text=text,
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

def test_evidence_group_is_covered_by_matching_chunk():
    group = ExpectedEvidenceGroup(
        group_id="progression_stages",
        description=(
            "Subject failure triggers staged progression."
        ),
        alternatives=(
            ExpectedEvidence(
                policy_id="220",
                heading_path=(
                    "Section 6 - Procedures",
                    (
                        "Part A - Monitoring and "
                        "Determining Academic Progression"
                    ),
                ),
                text_contains=(
                    "three stages of academic progression"
                ),
            ),
        ),
    )

    chunks = (
        make_chunk(
            policy_id="220",
            heading_path=(
                "Section 6 - Procedures",
                (
                    "Part A - Monitoring and "
                    "Determining Academic Progression"
                ),
            ),
            text=(
                "Students who experience subject failure "
                "will trigger one of three stages of "
                "academic progression."
            ),
        ),
    )

    assert is_expected_evidence_group_covered(
        chunks,
        group,
    )

def test_evidence_group_accepts_any_alternative():
    group = ExpectedEvidenceGroup(
        group_id="support_and_interventions",
        description=(
            "Progression includes support."
        ),
        alternatives=(
            ExpectedEvidence(
                policy_id="220",
                heading_path=(
                    "Section 5 - Policy Statement",
                ),
                text_contains=(
                    "academic and non-academic support"
                ),
            ),
            ExpectedEvidence(
                policy_id="220",
                heading_path=(
                    "Section 6 - Procedures",
                    (
                        "Part A - Monitoring and "
                        "Determining Academic Progression"
                    ),
                ),
                text_contains=(
                    "associated support and interventions"
                ),
            ),
        ),
    )

    chunks = (
        make_chunk(
            policy_id="220",
            heading_path=(
                "Section 6 - Procedures",
                (
                    "Part A - Monitoring and "
                    "Determining Academic Progression"
                ),
            ),
            text=(
                "Each stage has associated support "
                "and interventions."
            ),
        ),
    )

    assert is_expected_evidence_group_covered(
        chunks,
        group,
    )

def test_evidence_group_can_be_covered_by_later_chunk():
    group = ExpectedEvidenceGroup(
        group_id="support_mechanisms",
        description=(
            "Admission support mechanisms are provided."
        ),
        alternatives=(
            ExpectedEvidence(
                policy_id="169",
                heading_path=(
                    "Section 5 - Policy Statement",
                ),
                text_contains=(
                    "special entry access schemes"
                ),
            ),
        ),
    )

    chunks = (
        make_chunk(
            policy_id="169",
            heading_path=(
                "Section 5 - Policy Statement",
            ),
            text=(
                "The University supports participation."
            ),
        ),
        make_chunk(
            policy_id="169",
            heading_path=(
                "Section 5 - Policy Statement",
            ),
            text=(
                "Support includes special entry access "
                "schemes and alternative entry programs."
            ),
        ),
    )

    assert is_expected_evidence_group_covered(
        chunks,
        group,
    )

def test_evidence_group_is_not_covered_when_no_chunk_matches():
    group = ExpectedEvidenceGroup(
        group_id="support_mechanisms",
        description=(
            "Admission support mechanisms are provided."
        ),
        alternatives=(
            ExpectedEvidence(
                policy_id="169",
                heading_path=(
                    "Section 5 - Policy Statement",
                ),
                text_contains=(
                    "special entry access schemes"
                ),
            ),
        ),
    )

    chunks = (
        make_chunk(
            policy_id="169",
            heading_path=(
                "Section 5 - Policy Statement",
            ),
            text=(
                "This text does not contain the "
                "required mechanism."
            ),
        ),
    )

    assert not is_expected_evidence_group_covered(
        chunks,
        group,
    )

def test_evidence_group_is_not_covered_by_empty_chunks():
    group = ExpectedEvidenceGroup(
        group_id="support_mechanisms",
        description=(
            "Admission support mechanisms are provided."
        ),
        alternatives=(
            ExpectedEvidence(
                policy_id="169",
                heading_path=(
                    "Section 5 - Policy Statement",
                ),
            ),
        ),
    )

    assert not is_expected_evidence_group_covered(
        (),
        group,
    )

def test_evidence_group_preserves_expected_evidence_matching_rules():
    group = ExpectedEvidenceGroup(
        group_id="entry_requirements",
        description=(
            "Applicants meet admission requirements."
        ),
        alternatives=(
            ExpectedEvidence(
                policy_id="340",
                heading_path=(
                    "Section 6 - Procedures",
                    "Part A - Entry Criteria",
                ),
                text_contains=(
                    "General Admission Requirements"
                ),
            ),
        ),
    )

    wrong_heading = make_chunk(
        policy_id="340",
        heading_path=(
            "Section 5 - Policy Statement",
        ),
        text=(
            "General Admission Requirements"
        ),
    )

    assert not is_expected_evidence_group_covered(
        (wrong_heading,),
        group,
    )
