import pytest
from backend.routing.material_propositions import (
    MaterialProposition,
    MaterialPropositionKind,
)
from backend.routing.material_requirements import (
    MaterialRequirement,
    MaterialRequirementKind,
)
from backend.routing.proposition_coverage import (
    PropositionCoverageStatus,
    PropositionEvidenceCoverage,
    QuestionEvidenceCoverage,
)

def _proposition(
    relation: str,
) -> MaterialProposition:
    return MaterialProposition(
        kind=(
            MaterialPropositionKind.RELATIONAL
        ),
        relations=(
            MaterialRequirement(
                kind=(
                    MaterialRequirementKind.RELATION
                ),
                text=relation,
            ),
        ),
    )

def test_covered_proposition_preserves_supporting_evidence():
    coverage = PropositionEvidenceCoverage(
        proposition=_proposition(
            "provide",
        ),
        status=(
            PropositionCoverageStatus.COVERED
        ),
        score=0.95,
        evidence_indexes=(
            0,
            2,
        ),
    )

    assert coverage.score == 0.95

    assert coverage.evidence_indexes == (
        0,
        2,
    )

def test_covered_proposition_requires_supporting_evidence():
    with pytest.raises(
        ValueError,
        match=(
            "Covered propositions must "
            "identify supporting evidence."
        ),
    ):
        PropositionEvidenceCoverage(
            proposition=_proposition(
                "provide",
            ),
            status=(
                PropositionCoverageStatus.COVERED
            ),
            score=0.95,
        )

def test_coverage_score_must_be_bounded():
    with pytest.raises(
        ValueError,
        match=(
            "Coverage score must be between "
            "0.0 and 1.0."
        ),
    ):
        PropositionEvidenceCoverage(
            proposition=_proposition(
                "provide",
            ),
            status=(
                PropositionCoverageStatus.UNCOVERED
            ),
            score=1.01,
        )

def test_question_reports_all_propositions_covered():
    result = QuestionEvidenceCoverage(
        propositions=(
            PropositionEvidenceCoverage(
                proposition=_proposition(
                    "provide",
                ),
                status=(
                    PropositionCoverageStatus.COVERED
                ),
                score=0.97,
                evidence_indexes=(
                    0,
                ),
            ),
            PropositionEvidenceCoverage(
                proposition=_proposition(
                    "allow",
                ),
                status=(
                    PropositionCoverageStatus.COVERED
                ),
                score=0.91,
                evidence_indexes=(
                    1,
                ),
            ),
        )
    )

    assert result.all_covered is True
    assert result.minimum_score == 0.91

def test_question_is_not_fully_covered_when_one_proposition_is_partial():
    result = QuestionEvidenceCoverage(
        propositions=(
            PropositionEvidenceCoverage(
                proposition=_proposition(
                    "provide",
                ),
                status=(
                    PropositionCoverageStatus.COVERED
                ),
                score=0.96,
                evidence_indexes=(
                    0,
                ),
            ),
            PropositionEvidenceCoverage(
                proposition=_proposition(
                    "allow",
                ),
                status=(
                    PropositionCoverageStatus.PARTIAL
                ),
                score=0.55,
                evidence_indexes=(
                    1,
                ),
            ),
        )
    )

    assert result.all_covered is False
    assert result.minimum_score == 0.55

def test_question_coverage_cannot_be_empty():
    with pytest.raises(
        ValueError,
        match=(
            "Question evidence coverage "
            "cannot be empty."
        ),
    ):
        QuestionEvidenceCoverage(
            propositions=()
        )