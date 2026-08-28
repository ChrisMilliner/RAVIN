import pytest
from backend.routing.answerability import (
    AnswerabilityResult,
)
from backend.routing.material_propositions import (
    MaterialProposition,
    MaterialPropositionKind,
    MaterialQuestionPropositions,
)
from backend.routing.material_requirements import (
    MaterialRequirement,
    MaterialRequirementKind,
)
from backend.routing.proposition_coverage import (
    PropositionCoverageStatus,
)
from backend.routing.proposition_coverage_assessor import (
    PropositionCoverageAssessor,
)

class FakeAnswerabilityProvider:
    def __init__(
        self,
        results: tuple[
            AnswerabilityResult,
            ...
        ],
    ) -> None:
        self._results = list(
            results
        )

        self.questions: list[
            str
        ] = []

    def score(
        self,
        question: str,
        evidence_texts: tuple[
            str,
            ...
        ],
    ) -> AnswerabilityResult:
        self.questions.append(
            question
        )

        return self._results.pop(
            0
        )

def _requirement(
    kind: MaterialRequirementKind,
    text: str,
) -> MaterialRequirement:
    return MaterialRequirement(
        kind=kind,
        text=text,
    )

def _proposition(
    requested_attribute: str,
) -> MaterialProposition:
    return MaterialProposition(
        kind=(
            MaterialPropositionKind.RELATIONAL
        ),
        subjects=(
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "staff",
            ),
        ),
        relations=(
            _requirement(
                MaterialRequirementKind.RELATION,
                "receive",
            ),
        ),
        requested_attributes=(
            _requirement(
                MaterialRequirementKind.REQUESTED_ATTRIBUTE,
                requested_attribute,
            ),
        ),
    )

def test_marks_strong_proposition_as_covered():
    provider = FakeAnswerabilityProvider(
        results=(
            AnswerabilityResult(
                scores=(
                    0.95,
                    0.30,
                )
            ),
        )
    )

    assessor = PropositionCoverageAssessor(
        provider,
        covered_threshold=0.80,
        partial_threshold=0.40,
    )

    result = assessor.assess(
        "What benefits do staff receive?",
        MaterialQuestionPropositions(
            propositions=(
                _proposition(
                    "employment benefits",
                ),
            )
        ),
        (
            "Staff receive employment benefits.",
            "Unrelated evidence.",
        ),
    )

    coverage = result.propositions[0]

    assert (
        coverage.status
        == PropositionCoverageStatus.COVERED
    )

    assert coverage.score == 0.95

    assert coverage.evidence_indexes == (
        0,
    )

def test_marks_middle_score_as_partial():
    provider = FakeAnswerabilityProvider(
        results=(
            AnswerabilityResult(
                scores=(
                    0.25,
                    0.60,
                )
            ),
        )
    )

    assessor = PropositionCoverageAssessor(
        provider,
        covered_threshold=0.80,
        partial_threshold=0.40,
    )

    result = assessor.assess(
        "What benefits do staff receive?",
        MaterialQuestionPropositions(
            propositions=(
                _proposition(
                    "employment benefits",
                ),
            )
        ),
        (
            "Weak evidence.",
            "Partially relevant evidence.",
        ),
    )

    coverage = result.propositions[0]

    assert (
        coverage.status
        == PropositionCoverageStatus.PARTIAL
    )

    assert coverage.evidence_indexes == (
        1,
    )

def test_marks_weak_proposition_as_uncovered():
    provider = FakeAnswerabilityProvider(
        results=(
            AnswerabilityResult(
                scores=(
                    0.10,
                    0.20,
                )
            ),
        )
    )

    assessor = PropositionCoverageAssessor(
        provider,
        covered_threshold=0.80,
        partial_threshold=0.40,
    )

    result = assessor.assess(
        "What benefits do staff receive?",
        MaterialQuestionPropositions(
            propositions=(
                _proposition(
                    "employment benefits",
                ),
            )
        ),
        (
            "Unrelated evidence one.",
            "Unrelated evidence two.",
        ),
    )

    coverage = result.propositions[0]

    assert (
        coverage.status
        == PropositionCoverageStatus.UNCOVERED
    )

    assert coverage.evidence_indexes == ()

def test_scores_multiple_propositions_separately():
    provider = FakeAnswerabilityProvider(
        results=(
            AnswerabilityResult(
                scores=(
                    0.92,
                )
            ),
            AnswerabilityResult(
                scores=(
                    0.15,
                )
            ),
        )
    )

    assessor = PropositionCoverageAssessor(
        provider,
        covered_threshold=0.80,
        partial_threshold=0.40,
    )

    result = assessor.assess(
        (
            "What employment benefits and salary "
            "changes do staff receive?"
        ),
        MaterialQuestionPropositions(
            propositions=(
                _proposition(
                    "employment benefits",
                ),
                _proposition(
                    "salary changes",
                ),
            )
        ),
        (
            "Evidence text.",
        ),
    )

    assert len(
        provider.questions
    ) == 2

    assert (
        "employment benefits"
        in provider.questions[0]
    )

    assert (
        "salary changes"
        in provider.questions[1]
    )

    assert (
        result.propositions[0].status
        == PropositionCoverageStatus.COVERED
    )

    assert (
        result.propositions[1].status
        == PropositionCoverageStatus.UNCOVERED
    )

    assert result.all_covered is False

def test_coverage_query_preserves_scope_and_qualifier():
    proposition = MaterialProposition(
        kind=(
            MaterialPropositionKind.RELATIONAL
        ),
        subjects=(
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "a student",
            ),
        ),
        scopes=(
            _requirement(
                MaterialRequirementKind.CONCEPT,
                "Academic Progression Stage Three",
            ),
        ),
        relations=(
            _requirement(
                MaterialRequirementKind.RELATION,
                "submit",
            ),
        ),
        qualifiers=(
            _requirement(
                MaterialRequirementKind.QUALIFIER,
                "normally",
            ),
        ),
    )

    provider = FakeAnswerabilityProvider(
        results=(
            AnswerabilityResult(
                scores=(
                    0.90,
                )
            ),
        )
    )

    assessor = PropositionCoverageAssessor(
        provider,
        covered_threshold=0.80,
        partial_threshold=0.40,
    )

    assessor.assess(
        "What must the student submit?",
        MaterialQuestionPropositions(
            propositions=(
                proposition,
            )
        ),
        (
            "Evidence.",
        ),
    )

    query = provider.questions[0]

    assert "a student" in query

    assert (
        "Academic Progression Stage Three"
        in query
    )

    assert "submit" in query
    assert "normally" in query

def test_rejects_invalid_threshold_order():
    with pytest.raises(
        ValueError,
        match=(
            "Coverage thresholds must satisfy "
            "0.0 <= partial < covered <= 1.0."
        ),
    ):
        PropositionCoverageAssessor(
            FakeAnswerabilityProvider(
                results=()
            ),
            covered_threshold=0.40,
            partial_threshold=0.80,
        )