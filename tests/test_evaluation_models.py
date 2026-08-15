import pytest

from backend.evaluation.models import (
    DEFAULT_TOP_1_PASS_THRESHOLD,
    DEFAULT_TOP_K,
    EvaluationConfig,
    EvaluationQuestion,
    ExpectedEvidence,
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