import pytest

from backend.evaluation.models import (
    DEFAULT_TOP_1_PASS_THRESHOLD,
    DEFAULT_TOP_K,
    EvaluationBehavior,
    EvaluationConfig,
    EvaluationQuestion,
    ExpectedEvidence,
    EvaluationPopulation,
    ExpectedEvidenceGroup,
    GroundedOverviewGroupResult,
    GroundedOverviewQuestionResult,
    GroundedOverviewEvaluationResult,
)

def make_expected_evidence() -> ExpectedEvidence:
    return ExpectedEvidence(
        policy_id="208",
        heading_path=(
            "Section 4 - Key Decisions",
        ),
    )

def test_expected_evidence_preserves_policy_and_heading_path():
    evidence = make_expected_evidence()

    assert evidence.policy_id == "208"
    assert evidence.heading_path == (
        "Section 4 - Key Decisions",
    )

def test_evaluation_question_supports_multiple_valid_evidence_locations():
    question = EvaluationQuestion(
        question_id="Q001",
        question="Who approves changes to academic dress?",
        expected_evidence=(
            ExpectedEvidence(
                policy_id="208",
                heading_path=(
                    "Section 4 - Key Decisions",
                ),
            ),
            ExpectedEvidence(
                policy_id="208",
                heading_path=(
                    "Section 6 - Procedures",
                    (
                        "Part B - Requests for Changes or "
                        "Additions to Academic Dress"
                    ),
                ),
            ),
        ),
    )

    assert len(question.expected_evidence) == 2

def test_evaluation_question_defaults_to_direct_answer_behavior():
    question = EvaluationQuestion(
        question_id="Q001",
        question="Who approves changes to academic dress?",
        expected_evidence=(
            make_expected_evidence(),
        ),
    )

    assert (
        question.behavior
        == EvaluationBehavior.DIRECT_ANSWER
    )

def test_evaluation_question_preserves_explicit_behavior():
    question = EvaluationQuestion(
        question_id="Q001",
        question="What admission requirements apply to me?",
        expected_evidence=(
            make_expected_evidence(),
        ),
        behavior=EvaluationBehavior.CLARIFY,
    )

    assert (
        question.behavior
        == EvaluationBehavior.CLARIFY
    )

def test_evaluation_behavior_defines_supported_response_types():
    assert {
        behavior.value
        for behavior in EvaluationBehavior
    } == {
        "direct_answer",
        "grounded_overview",
        "clarify",
        "no_grounded_answer",
    }

def test_evaluation_config_uses_project_quality_gate_by_default():
    config = EvaluationConfig()

    assert config.top_k == DEFAULT_TOP_K
    assert config.top_1_pass_threshold == 0.95
    assert (
        config.top_1_pass_threshold
        == DEFAULT_TOP_1_PASS_THRESHOLD
    )

def test_evaluation_config_rejects_invalid_top_k():
    with pytest.raises(
        ValueError,
        match="Evaluation top_k must be greater than zero.",
    ):
        EvaluationConfig(top_k=0)

def test_evaluation_config_rejects_threshold_above_one():
    with pytest.raises(
        ValueError,
        match="Top-1 pass threshold must be between 0 and 1.",
    ):
        EvaluationConfig(
            top_1_pass_threshold=1.01,
        )

def test_evaluation_question_rejects_empty_question():
    with pytest.raises(
        ValueError,
        match="Evaluation question cannot be empty.",
    ):
        EvaluationQuestion(
            question_id="Q001",
            question="   ",
            expected_evidence=(
                make_expected_evidence(),
            ),
        )

def test_evaluation_question_requires_expected_evidence():
    with pytest.raises(
        ValueError,
        match=(
            "Evaluation question must define expected evidence."
        ),
    ):
        EvaluationQuestion(
            question_id="Q001",
            question="Who approves changes to academic dress?",
            expected_evidence=(),
        )

def test_no_grounded_answer_allows_empty_expected_evidence():
    question = EvaluationQuestion(
        question_id="Q099",
        question="What is the university president's favourite restaurant?",
        expected_evidence=(),
        behavior=(
            EvaluationBehavior.NO_GROUNDED_ANSWER
        ),
    )

    assert question.expected_evidence == ()
    assert (
        question.behavior
        == EvaluationBehavior.NO_GROUNDED_ANSWER
    )


def test_no_grounded_answer_rejects_expected_evidence():
    with pytest.raises(
        ValueError,
        match=(
            "No-grounded-answer questions "
            "must not define expected evidence."
        ),
    ):
        EvaluationQuestion(
            question_id="Q099",
            question="Unsupported question",
            expected_evidence=(
                make_expected_evidence(),
            ),
            behavior=(
                EvaluationBehavior.NO_GROUNDED_ANSWER
            ),
        )


@pytest.mark.parametrize(
    "behavior",
    (
        EvaluationBehavior.DIRECT_ANSWER,
        EvaluationBehavior.GROUNDED_OVERVIEW,
        EvaluationBehavior.CLARIFY,
    ),
)
def test_grounded_behaviors_require_expected_evidence(
    behavior,
):
    with pytest.raises(
        ValueError,
        match=(
            "Evaluation question must define "
            "expected evidence."
        ),
    ):
        EvaluationQuestion(
            question_id="Q001",
            question="Test question",
            expected_evidence=(),
            behavior=behavior,
        )

def test_expected_evidence_preserves_matching_controls():
    expected = ExpectedEvidence(
        policy_id="208",
        heading_path=(
            "Section 6 - Procedures",
        ),
        allow_descendants=True,
        text_contains="answer-bearing evidence",
    )

    assert expected.allow_descendants is True
    assert (
        expected.text_contains
        == "answer-bearing evidence"
    )

def test_expected_evidence_rejects_empty_text_fragment():
    with pytest.raises(
        ValueError,
        match=(
            "Expected evidence text fragment "
            "cannot be empty."
        ),
    ):
        ExpectedEvidence(
            policy_id="208",
            heading_path=(
                "Section 6 - Procedures",
            ),
            text_contains="   ",
        )

def test_evaluation_question_rejects_invalid_behavior_type():
    with pytest.raises(
        ValueError,
        match=(
            "Evaluation question behavior must be "
            "an EvaluationBehavior."
        ),
    ):
        EvaluationQuestion(
            question_id="Q001",
            question="Test question",
            expected_evidence=(
                make_expected_evidence(),
            ),
            behavior="clarify",  # type: ignore[arg-type]
        )

def test_evaluation_population_preserves_behavior_counts():
    population = EvaluationPopulation(
        dataset_questions=30,
        direct_answer_questions=26,
        grounded_overview_questions=4,
        clarify_questions=0,
        no_grounded_answer_questions=0,
    )

    assert population.dataset_questions == 30
    assert population.direct_answer_questions == 26
    assert population.grounded_overview_questions == 4
    assert population.clarify_questions == 0
    assert population.no_grounded_answer_questions == 0

def test_evaluation_population_rejects_mismatched_counts():
    with pytest.raises(
        ValueError,
        match=(
            "Evaluation population behavior counts "
            "must equal the dataset question count."
        ),
    ):
        EvaluationPopulation(
            dataset_questions=30,
            direct_answer_questions=25,
            grounded_overview_questions=4,
            clarify_questions=0,
            no_grounded_answer_questions=0,
        )

def test_evaluation_population_rejects_negative_counts():
    with pytest.raises(
        ValueError,
        match=(
            "Evaluation population counts "
            "cannot be negative."
        ),
    ):
        EvaluationPopulation(
            dataset_questions=30,
            direct_answer_questions=27,
            grounded_overview_questions=4,
            clarify_questions=-1,
            no_grounded_answer_questions=0,
        )

def test_expected_evidence_group_preserves_alternatives():
    evidence = make_expected_evidence()

    group = ExpectedEvidenceGroup(
        group_id="support_mechanisms",
        description=(
            "Mechanisms used to support participation."
        ),
        alternatives=(evidence,),
    )

    assert group.group_id == "support_mechanisms"
    assert (
        group.description
        == "Mechanisms used to support participation."
    )
    assert group.alternatives == (evidence,)


def test_expected_evidence_group_rejects_empty_id():
    with pytest.raises(
        ValueError,
        match=(
            "Expected evidence group ID "
            "cannot be empty."
        ),
    ):
        ExpectedEvidenceGroup(
            group_id="   ",
            description="Test concept.",
            alternatives=(
                make_expected_evidence(),
            ),
        )


def test_expected_evidence_group_rejects_empty_description():
    with pytest.raises(
        ValueError,
        match=(
            "Expected evidence group description "
            "cannot be empty."
        ),
    ):
        ExpectedEvidenceGroup(
            group_id="concept_one",
            description="   ",
            alternatives=(
                make_expected_evidence(),
            ),
        )


def test_expected_evidence_group_requires_alternative():
    with pytest.raises(
        ValueError,
        match=(
            "Expected evidence group must define "
            "at least one evidence alternative."
        ),
    ):
        ExpectedEvidenceGroup(
            group_id="concept_one",
            description="Test concept.",
            alternatives=(),
        )

def test_expected_evidence_group_rejects_invalid_alternative_type():
    with pytest.raises(
        ValueError,
        match=(
            "Expected evidence group alternatives "
            "must be ExpectedEvidence values."
        ),
    ):
        ExpectedEvidenceGroup(
            group_id="concept_one",
            description="Test concept.",
            alternatives=(
                "invalid",  # type: ignore[arg-type]
            ),
        )

def test_evaluation_question_defaults_evidence_groups_to_empty():
    question = EvaluationQuestion(
        question_id="Q001",
        question="Test question",
        expected_evidence=(
            make_expected_evidence(),
        ),
    )

    assert question.expected_evidence_groups == ()

def test_evaluation_question_preserves_evidence_groups():
    group = ExpectedEvidenceGroup(
        group_id="concept_one",
        description="Required overview concept.",
        alternatives=(
            make_expected_evidence(),
        ),
    )

    question = EvaluationQuestion(
        question_id="Q001",
        question="Test question",
        expected_evidence=(
            make_expected_evidence(),
        ),
        expected_evidence_groups=(group,),
    )

    assert question.expected_evidence_groups == (
        group,
    )

def test_grounded_overview_group_result_preserves_values():
    result = GroundedOverviewGroupResult(
        group_id="concept_one",
        covered=True,
    )

    assert result.group_id == "concept_one"
    assert result.covered is True

def test_grounded_overview_group_result_rejects_empty_id():
    with pytest.raises(
        ValueError,
        match=(
            "Grounded overview group result ID "
            "cannot be empty."
        ),
    ):
        GroundedOverviewGroupResult(
            group_id="   ",
            covered=True,
        )

def test_grounded_overview_question_calculates_complete_coverage():
    result = GroundedOverviewQuestionResult(
        question_id="RB002",
        group_results=(
            GroundedOverviewGroupResult(
                group_id="group_one",
                covered=True,
            ),
            GroundedOverviewGroupResult(
                group_id="group_two",
                covered=True,
            ),
        ),
    )

    assert result.total_groups == 2
    assert result.covered_groups == 2
    assert result.evidence_coverage == 1.0
    assert result.passed is True

def test_grounded_overview_question_fails_when_group_missing():
    result = GroundedOverviewQuestionResult(
        question_id="RB002",
        group_results=(
            GroundedOverviewGroupResult(
                group_id="group_one",
                covered=True,
            ),
            GroundedOverviewGroupResult(
                group_id="group_two",
                covered=False,
            ),
        ),
    )

    assert result.total_groups == 2
    assert result.covered_groups == 1
    assert result.evidence_coverage == 0.5
    assert result.passed is False

def test_grounded_overview_question_rejects_empty_groups():
    with pytest.raises(
        ValueError,
        match=(
            "Grounded overview question result "
            "must contain at least one group result."
        ),
    ):
        GroundedOverviewQuestionResult(
            question_id="RB002",
            group_results=(),
        )

def test_grounded_overview_question_rejects_duplicate_groups():
    with pytest.raises(
        ValueError,
        match=(
            "Grounded overview question result "
            "cannot contain duplicate group IDs."
        ),
    ):
        GroundedOverviewQuestionResult(
            question_id="RB002",
            group_results=(
                GroundedOverviewGroupResult(
                    group_id="same_group",
                    covered=True,
                ),
                GroundedOverviewGroupResult(
                    group_id="same_group",
                    covered=False,
                ),
            ),
        )

def test_grounded_overview_evaluation_calculates_question_gate():
    evaluation = GroundedOverviewEvaluationResult(
        question_results=(
            GroundedOverviewQuestionResult(
                question_id="RB002",
                group_results=(
                    GroundedOverviewGroupResult(
                        group_id="one",
                        covered=True,
                    ),
                ),
            ),
            GroundedOverviewQuestionResult(
                question_id="RB003",
                group_results=(
                    GroundedOverviewGroupResult(
                        group_id="two",
                        covered=False,
                    ),
                ),
            ),
        ),
        pass_threshold=0.95,
    )

    assert evaluation.total_questions == 2
    assert evaluation.passed_questions == 1
    assert evaluation.question_pass_rate == 0.5

    assert evaluation.total_groups == 2
    assert evaluation.covered_groups == 1
    assert (
        evaluation.evidence_group_coverage
        == 0.5
    )

    assert evaluation.passed is False

def test_grounded_overview_evaluation_passes_threshold():
    evaluation = GroundedOverviewEvaluationResult(
        question_results=(
            GroundedOverviewQuestionResult(
                question_id="RB002",
                group_results=(
                    GroundedOverviewGroupResult(
                        group_id="one",
                        covered=True,
                    ),
                ),
            ),
        ),
        pass_threshold=0.95,
    )

    assert evaluation.question_pass_rate == 1.0
    assert evaluation.passed is True

def test_grounded_overview_evaluation_rejects_invalid_threshold():
    with pytest.raises(
        ValueError,
        match=(
            "Grounded overview pass threshold "
            "must be between 0 and 1."
        ),
    ):
        GroundedOverviewEvaluationResult(
            question_results=(
                GroundedOverviewQuestionResult(
                    question_id="RB002",
                    group_results=(
                        GroundedOverviewGroupResult(
                            group_id="one",
                            covered=True,
                        ),
                    ),
                ),
            ),
            pass_threshold=1.01,
        )

def test_grounded_overview_group_result_rejects_invalid_covered_type():
    with pytest.raises(
        ValueError,
        match=(
            "Grounded overview group result "
            "covered must be a boolean."
        ),
    ):
        GroundedOverviewGroupResult(
            group_id="concept_one",
            covered="yes",  # type: ignore[arg-type]
        )

def test_grounded_overview_question_rejects_empty_question_id():
    with pytest.raises(
        ValueError,
        match=(
            "Grounded overview question result ID "
            "cannot be empty."
        ),
    ):
        GroundedOverviewQuestionResult(
            question_id="   ",
            group_results=(
                GroundedOverviewGroupResult(
                    group_id="one",
                    covered=True,
                ),
            ),
        )

def test_grounded_overview_question_rejects_invalid_group_result_type():
    with pytest.raises(
        ValueError,
        match=(
            "Grounded overview question group "
            "results must be "
            "GroundedOverviewGroupResult values."
        ),
    ):
        GroundedOverviewQuestionResult(
            question_id="RB002",
            group_results=(
                "invalid",  # type: ignore[arg-type]
            ),
        )

def test_grounded_overview_evaluation_rejects_empty_results():
    with pytest.raises(
        ValueError,
        match=(
            "Grounded overview evaluation "
            "requires at least one question result."
        ),
    ):
        GroundedOverviewEvaluationResult(
            question_results=(),
            pass_threshold=0.95,
        )

def test_grounded_overview_evaluation_rejects_invalid_result_type():
    with pytest.raises(
        ValueError,
        match=(
            "Grounded overview evaluation question "
            "results must be "
            "GroundedOverviewQuestionResult values."
        ),
    ):
        GroundedOverviewEvaluationResult(
            question_results=(
                "invalid",  # type: ignore[arg-type]
            ),
            pass_threshold=0.95,
        )

def test_grounded_overview_evaluation_rejects_duplicate_questions():
    question = GroundedOverviewQuestionResult(
        question_id="RB002",
        group_results=(
            GroundedOverviewGroupResult(
                group_id="one",
                covered=True,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Grounded overview evaluation cannot "
            "contain duplicate question IDs."
        ),
    ):
        GroundedOverviewEvaluationResult(
            question_results=(
                question,
                question,
            ),
            pass_threshold=0.95,
        )
