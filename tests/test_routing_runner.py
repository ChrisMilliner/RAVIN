from typing import cast
import pytest
from backend.behavior import AnswerBehavior
from backend.evaluation.routing_dataset import (
    load_routing_evaluation_questions,
)
from backend.evaluation.routing_models import (
    RoutingEvaluationConfig,
    RoutingEvaluationQuestion,
    RoutingPrediction,
)
from backend.evaluation.routing_runner import (
    run_routing_evaluation,
)
from backend.routing.models import (
    EvidenceSufficiency,
    QuestionIntent,
)

def _questions() -> tuple[
    RoutingEvaluationQuestion,
    ...,
]:
    return (
        RoutingEvaluationQuestion(
            question_id="RQ001",
            question="Focused supported question?",
            expected_intent=(
                QuestionIntent.FOCUSED
            ),
            expected_sufficiency=(
                EvidenceSufficiency.SUFFICIENT
            ),
            expected_behavior=(
                AnswerBehavior.DIRECT_ANSWER
            ),
        ),
        RoutingEvaluationQuestion(
            question_id="RQ002",
            question="Broad supported question?",
            expected_intent=QuestionIntent.BROAD,
            expected_sufficiency=(
                EvidenceSufficiency.SUFFICIENT
            ),
            expected_behavior=(
                AnswerBehavior.GROUNDED_OVERVIEW
            ),
        ),
        RoutingEvaluationQuestion(
            question_id="RQ003",
            question="Ambiguous question?",
            expected_intent=(
                QuestionIntent.AMBIGUOUS
            ),
            expected_sufficiency=None,
            expected_behavior=(
                AnswerBehavior.CLARIFY
            ),
        ),
        RoutingEvaluationQuestion(
            question_id="RQ004",
            question="Focused unsupported question?",
            expected_intent=(
                QuestionIntent.FOCUSED
            ),
            expected_sufficiency=(
                EvidenceSufficiency.INSUFFICIENT
            ),
            expected_behavior=(
                AnswerBehavior.NO_GROUNDED_ANSWER
            ),
        ),
        RoutingEvaluationQuestion(
            question_id="RQ005",
            question="Broad unsupported question?",
            expected_intent=QuestionIntent.BROAD,
            expected_sufficiency=(
                EvidenceSufficiency.INSUFFICIENT
            ),
            expected_behavior=(
                AnswerBehavior.NO_GROUNDED_ANSWER
            ),
        ),
    )

def _oracle_prediction(
    question: RoutingEvaluationQuestion,
) -> RoutingPrediction:
    return RoutingPrediction(
        question_id=question.question_id,
        predicted_intent=(
            question.expected_intent
        ),
        predicted_sufficiency=(
            question.expected_sufficiency
        ),
        predicted_behavior=(
            question.expected_behavior
        ),
    )

def test_runner_scores_perfect_predictions():
    result = run_routing_evaluation(
        _questions(),
        _oracle_prediction,
        RoutingEvaluationConfig(),
    )

    assert result.intent_result.overall_accuracy == 1.0
    assert result.intent_result.macro_accuracy == 1.0

    assert (
        result.sufficiency_result.overall_accuracy
        == 1.0
    )

    assert (
        result.sufficiency_result.macro_accuracy
        == 1.0
    )

    assert (
        result.behavior_result.overall_accuracy
        == 1.0
    )

    assert (
        result.behavior_result.macro_accuracy
        == 1.0
    )

    assert result.passed

def test_runner_records_expected_class_support():
    result = run_routing_evaluation(
        _questions(),
        _oracle_prediction,
        RoutingEvaluationConfig(),
    )

    intent_support = {
        class_result.label: class_result.support
        for class_result
        in result.intent_result.class_results
    }

    sufficiency_support = {
        class_result.label: class_result.support
        for class_result
        in result.sufficiency_result.class_results
    }

    behavior_support = {
        class_result.label: class_result.support
        for class_result
        in result.behavior_result.class_results
    }

    assert intent_support == {
        "focused": 2,
        "broad": 2,
        "ambiguous": 1,
    }

    assert sufficiency_support == {
        "sufficient": 2,
        "insufficient": 2,
    }

    assert behavior_support == {
        "direct_answer": 1,
        "grounded_overview": 1,
        "clarify": 1,
        "no_grounded_answer": 2,
    }

def test_runner_scores_incorrect_prediction():
    def predict(
        question: RoutingEvaluationQuestion,
    ) -> RoutingPrediction:
        if question.question_id == "RQ001":
            return RoutingPrediction(
                question_id=question.question_id,
                predicted_intent=(
                    QuestionIntent.BROAD
                ),
                predicted_sufficiency=(
                    EvidenceSufficiency.SUFFICIENT
                ),
                predicted_behavior=(
                    AnswerBehavior.GROUNDED_OVERVIEW
                ),
            )

        return _oracle_prediction(
            question
        )

    result = run_routing_evaluation(
        _questions(),
        predict,
        RoutingEvaluationConfig(),
    )

    assert (
        result.intent_result.overall_accuracy
        == 0.8
    )

    assert result.intent_result.macro_accuracy == (
        pytest.approx(
            (
                0.5
                + 1.0
                + 1.0
            )
            / 3
        )
    )

    assert (
        result.sufficiency_result.overall_accuracy
        == 1.0
    )

    assert (
        result.behavior_result.overall_accuracy
        == 0.8
    )

    assert result.behavior_result.macro_accuracy == (
        pytest.approx(
            (
                0.0
                + 1.0
                + 1.0
                + 1.0
            )
            / 4
        )
    )

    assert not result.passed

def test_runner_counts_clear_abstention_as_incorrect():
    def predict(
        question: RoutingEvaluationQuestion,
    ) -> RoutingPrediction:
        prediction = _oracle_prediction(
            question
        )

        if question.question_id != "RQ001":
            return prediction

        return RoutingPrediction(
            question_id=question.question_id,
            predicted_intent=(
                prediction.predicted_intent
            ),
            predicted_sufficiency=None,
            predicted_behavior=(
                prediction.predicted_behavior
            ),
        )

    result = run_routing_evaluation(
        _questions(),
        predict,
        RoutingEvaluationConfig(),
    )

    assert (
        result.sufficiency_result.overall_accuracy
        == 0.75
    )

    assert (
        result.sufficiency_result.macro_accuracy
        == 0.75
    )

    assert not result.sufficiency_result.passed
    assert not result.passed

def test_runner_excludes_ambiguous_truth_from_sufficiency_metric():
    def predict(
        question: RoutingEvaluationQuestion,
    ) -> RoutingPrediction:
        prediction = _oracle_prediction(
            question
        )

        if question.question_id != "RQ003":
            return prediction

        return RoutingPrediction(
            question_id=question.question_id,
            predicted_intent=(
                QuestionIntent.AMBIGUOUS
            ),
            predicted_sufficiency=(
                EvidenceSufficiency.SUFFICIENT
            ),
            predicted_behavior=(
                AnswerBehavior.CLARIFY
            ),
        )

    result = run_routing_evaluation(
        _questions(),
        predict,
        RoutingEvaluationConfig(),
    )

    assert (
        result.sufficiency_result.overall_accuracy
        == 1.0
    )

    assert (
        result.sufficiency_result.macro_accuracy
        == 1.0
    )

    assert result.passed

def test_runner_rejects_empty_question_set():
    with pytest.raises(
        ValueError,
        match=(
            "Cannot evaluate an empty routing "
            "question set."
        ),
    ):
        run_routing_evaluation(
            (),
            _oracle_prediction,
            RoutingEvaluationConfig(),
        )

def test_runner_rejects_invalid_prediction():
    invalid_prediction = cast(
        RoutingPrediction,
        "invalid",
    )

    def predict(
        question: RoutingEvaluationQuestion,
    ) -> RoutingPrediction:
        return invalid_prediction

    with pytest.raises(
        ValueError,
        match=(
            "Routing prediction function must "
            "return a RoutingPrediction."
        ),
    ):
        run_routing_evaluation(
            _questions(),
            predict,
            RoutingEvaluationConfig(),
        )

def test_runner_rejects_mismatched_prediction_id():
    def predict(
        question: RoutingEvaluationQuestion,
    ) -> RoutingPrediction:
        return RoutingPrediction(
            question_id="WRONG-ID",
            predicted_intent=(
                question.expected_intent
            ),
            predicted_sufficiency=(
                question.expected_sufficiency
            ),
            predicted_behavior=(
                question.expected_behavior
            ),
        )

    with pytest.raises(
        ValueError,
        match=(
            "Routing prediction question ID "
            "must match the evaluation "
            "question ID."
        ),
    ):
        run_routing_evaluation(
            _questions(),
            predict,
            RoutingEvaluationConfig(),
        )

def test_real_routing_baseline_runs_with_oracle_predictions():
    questions = (
        load_routing_evaluation_questions(
            "evaluation/routing_baseline.json"
        )
    )

    result = run_routing_evaluation(
        questions,
        _oracle_prediction,
        RoutingEvaluationConfig(),
    )

    assert len(result.question_results) == 50

    intent_support = {
        class_result.label: class_result.support
        for class_result
        in result.intent_result.class_results
    }

    sufficiency_support = {
        class_result.label: class_result.support
        for class_result
        in result.sufficiency_result.class_results
    }

    behavior_support = {
        class_result.label: class_result.support
        for class_result
        in result.behavior_result.class_results
    }

    assert intent_support == {
        "focused": 32,
        "broad": 8,
        "ambiguous": 10,
    }

    assert sufficiency_support == {
        "sufficient": 30,
        "insufficient": 10,
    }

    assert behavior_support == {
        "direct_answer": 26,
        "grounded_overview": 4,
        "clarify": 10,
        "no_grounded_answer": 10,
    }

    assert result.intent_result.macro_accuracy == 1.0

    assert (
        result.sufficiency_result.macro_accuracy
        == 1.0
    )

    assert (
        result.behavior_result.macro_accuracy
        == 1.0
    )

    assert result.passed