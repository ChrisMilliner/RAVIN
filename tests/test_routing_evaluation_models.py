import pytest
from backend.behavior import AnswerBehavior
from backend.evaluation.routing_models import (
    RoutingEvaluationQuestion,
)
from backend.routing.models import (
    EvidenceSufficiency,
    QuestionIntent,
)

def test_focused_sufficient_question_is_valid():
    question = RoutingEvaluationQuestion(
        question_id="RQ001",
        question=(
            "Who approves changes to academic dress?"
        ),
        expected_intent=QuestionIntent.FOCUSED,
        expected_sufficiency=(
            EvidenceSufficiency.SUFFICIENT
        ),
        expected_behavior=(
            AnswerBehavior.DIRECT_ANSWER
        ),
    )

    assert question.expected_intent is (
        QuestionIntent.FOCUSED
    )

    assert question.expected_sufficiency is (
        EvidenceSufficiency.SUFFICIENT
    )

    assert question.expected_behavior is (
        AnswerBehavior.DIRECT_ANSWER
    )

def test_broad_sufficient_question_is_valid():
    question = RoutingEvaluationQuestion(
        question_id="RQ002",
        question=(
            "What happens when a student is not "
            "making satisfactory progress?"
        ),
        expected_intent=QuestionIntent.BROAD,
        expected_sufficiency=(
            EvidenceSufficiency.SUFFICIENT
        ),
        expected_behavior=(
            AnswerBehavior.GROUNDED_OVERVIEW
        ),
    )

    assert question.expected_behavior is (
        AnswerBehavior.GROUNDED_OVERVIEW
    )

def test_ambiguous_question_is_valid():
    question = RoutingEvaluationQuestion(
        question_id="RQ003",
        question="What do I need to submit?",
        expected_intent=(
            QuestionIntent.AMBIGUOUS
        ),
        expected_sufficiency=None,
        expected_behavior=(
            AnswerBehavior.CLARIFY
        ),
    )

    assert question.expected_sufficiency is None

    assert question.expected_behavior is (
        AnswerBehavior.CLARIFY
    )

def test_focused_insufficient_question_is_valid():
    question = RoutingEvaluationQuestion(
        question_id="RQ004",
        question=(
            "Does the policy guarantee a "
            "particular unsupported outcome?"
        ),
        expected_intent=QuestionIntent.FOCUSED,
        expected_sufficiency=(
            EvidenceSufficiency.INSUFFICIENT
        ),
        expected_behavior=(
            AnswerBehavior.NO_GROUNDED_ANSWER
        ),
    )

    assert question.expected_behavior is (
        AnswerBehavior.NO_GROUNDED_ANSWER
    )

def test_broad_insufficient_question_is_valid():
    question = RoutingEvaluationQuestion(
        question_id="RQ005",
        question=(
            "What support is guaranteed across "
            "an unsupported process?"
        ),
        expected_intent=QuestionIntent.BROAD,
        expected_sufficiency=(
            EvidenceSufficiency.INSUFFICIENT
        ),
        expected_behavior=(
            AnswerBehavior.NO_GROUNDED_ANSWER
        ),
    )

    assert question.expected_behavior is (
        AnswerBehavior.NO_GROUNDED_ANSWER
    )

def test_ambiguous_question_rejects_sufficiency_label():
    with pytest.raises(
        ValueError,
        match=(
            "Ambiguous routing questions must "
            "not define expected evidence "
            "sufficiency."
        ),
    ):
        RoutingEvaluationQuestion(
            question_id="RQ006",
            question="What do I need to submit?",
            expected_intent=(
                QuestionIntent.AMBIGUOUS
            ),
            expected_sufficiency=(
                EvidenceSufficiency.SUFFICIENT
            ),
            expected_behavior=(
                AnswerBehavior.CLARIFY
            ),
        )

def test_ambiguous_question_must_expect_clarify():
    with pytest.raises(
        ValueError,
        match=(
            "Ambiguous routing questions must "
            "expect clarify behavior."
        ),
    ):
        RoutingEvaluationQuestion(
            question_id="RQ007",
            question="What do I need to submit?",
            expected_intent=(
                QuestionIntent.AMBIGUOUS
            ),
            expected_sufficiency=None,
            expected_behavior=(
                AnswerBehavior.DIRECT_ANSWER
            ),
        )

def test_clear_question_requires_sufficiency_label():
    with pytest.raises(
        ValueError,
        match=(
            "Clear routing questions must "
            "define expected evidence "
            "sufficiency."
        ),
    ):
        RoutingEvaluationQuestion(
            question_id="RQ008",
            question=(
                "Who approves changes to "
                "academic dress?"
            ),
            expected_intent=(
                QuestionIntent.FOCUSED
            ),
            expected_sufficiency=None,
            expected_behavior=(
                AnswerBehavior.DIRECT_ANSWER
            ),
        )

def test_uncertain_cannot_be_evaluation_truth():
    with pytest.raises(
        ValueError,
        match=(
            "Uncertain is a runtime fallback "
            "and cannot be evaluation truth."
        ),
    ):
        RoutingEvaluationQuestion(
            question_id="RQ009",
            question="Who approves this?",
            expected_intent=(
                QuestionIntent.FOCUSED
            ),
            expected_sufficiency=(
                EvidenceSufficiency.UNCERTAIN
            ),
            expected_behavior=(
                AnswerBehavior.NO_GROUNDED_ANSWER
            ),
        )

def test_insufficient_evidence_requires_no_answer():
    with pytest.raises(
        ValueError,
        match=(
            "Insufficient evidence must "
            "expect no-grounded-answer "
            "behavior."
        ),
    ):
        RoutingEvaluationQuestion(
            question_id="RQ010",
            question="Who approves this?",
            expected_intent=(
                QuestionIntent.FOCUSED
            ),
            expected_sufficiency=(
                EvidenceSufficiency.INSUFFICIENT
            ),
            expected_behavior=(
                AnswerBehavior.DIRECT_ANSWER
            ),
        )

def test_focused_sufficient_question_rejects_wrong_behavior():
    with pytest.raises(
        ValueError,
        match=(
            "Expected behavior does not match "
            "the intent and evidence labels."
        ),
    ):
        RoutingEvaluationQuestion(
            question_id="RQ011",
            question="Who approves this?",
            expected_intent=(
                QuestionIntent.FOCUSED
            ),
            expected_sufficiency=(
                EvidenceSufficiency.SUFFICIENT
            ),
            expected_behavior=(
                AnswerBehavior.GROUNDED_OVERVIEW
            ),
        )

def test_broad_sufficient_question_rejects_wrong_behavior():
    with pytest.raises(
        ValueError,
        match=(
            "Expected behavior does not match "
            "the intent and evidence labels."
        ),
    ):
        RoutingEvaluationQuestion(
            question_id="RQ012",
            question="What happens in this process?",
            expected_intent=QuestionIntent.BROAD,
            expected_sufficiency=(
                EvidenceSufficiency.SUFFICIENT
            ),
            expected_behavior=(
                AnswerBehavior.DIRECT_ANSWER
            ),
        )