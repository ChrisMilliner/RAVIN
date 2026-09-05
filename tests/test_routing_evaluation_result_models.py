from typing import cast
import pytest
from backend.behavior import AnswerBehavior
from backend.evaluation.routing_models import (
    DEFAULT_ROUTING_PASS_THRESHOLD,
    RoutingClassResult,
    RoutingClassificationResult,
    RoutingEvaluationConfig,
    RoutingEvaluationRunResult,
    RoutingPrediction,
    RoutingQuestionEvaluationResult,
)
from backend.routing.models import (
    EvidenceSufficiency,
    QuestionIntent,
)

def _make_question_result(
    question_id: str = "RQ001",
) -> RoutingQuestionEvaluationResult:
    return RoutingQuestionEvaluationResult(
        question_id=question_id,
        expected_intent=QuestionIntent.FOCUSED,
        predicted_intent=QuestionIntent.FOCUSED,
        expected_sufficiency=(
            EvidenceSufficiency.SUFFICIENT
        ),
        predicted_sufficiency=(
            EvidenceSufficiency.SUFFICIENT
        ),
        expected_behavior=(
            AnswerBehavior.DIRECT_ANSWER
        ),
        predicted_behavior=(
            AnswerBehavior.DIRECT_ANSWER
        ),
    )

def _make_classification_result(
    label: str = "focused",
    accuracy: float = 1.0,
    threshold: float = 0.95,
) -> RoutingClassificationResult:
    support = 100
    correct = round(
        support * accuracy
    )

    return RoutingClassificationResult(
        overall_accuracy=accuracy,
        macro_accuracy=accuracy,
        class_results=(
            RoutingClassResult(
                label=label,
                support=support,
                correct=correct,
            ),
        ),
        pass_threshold=threshold,
    )

def test_routing_evaluation_config_uses_default_thresholds():
    config = RoutingEvaluationConfig()

    assert config.intent_pass_threshold == (
        DEFAULT_ROUTING_PASS_THRESHOLD
    )

    assert config.sufficiency_pass_threshold == (
        DEFAULT_ROUTING_PASS_THRESHOLD
    )

    assert config.behavior_pass_threshold == (
        DEFAULT_ROUTING_PASS_THRESHOLD
    )

def test_routing_evaluation_config_rejects_invalid_intent_threshold():
    with pytest.raises(
        ValueError,
        match=(
            "Routing evaluation thresholds "
            "must be between 0 and 1."
        ),
    ):
        RoutingEvaluationConfig(
            intent_pass_threshold=1.01,
        )

def test_routing_evaluation_config_rejects_invalid_sufficiency_threshold():
    with pytest.raises(
        ValueError,
        match=(
            "Routing evaluation thresholds "
            "must be between 0 and 1."
        ),
    ):
        RoutingEvaluationConfig(
            sufficiency_pass_threshold=-0.01,
        )

def test_routing_evaluation_config_rejects_invalid_behavior_threshold():
    with pytest.raises(
        ValueError,
        match=(
            "Routing evaluation thresholds "
            "must be between 0 and 1."
        ),
    ):
        RoutingEvaluationConfig(
            behavior_pass_threshold=1.01,
        )

def test_routing_prediction_accepts_valid_prediction():
    prediction = RoutingPrediction(
        question_id="RQ001",
        predicted_intent=QuestionIntent.FOCUSED,
        predicted_sufficiency=(
            EvidenceSufficiency.SUFFICIENT
        ),
        predicted_behavior=(
            AnswerBehavior.DIRECT_ANSWER
        ),
    )

    assert prediction.question_id == "RQ001"

    assert prediction.predicted_intent is (
        QuestionIntent.FOCUSED
    )

    assert prediction.predicted_sufficiency is (
        EvidenceSufficiency.SUFFICIENT
    )

    assert prediction.predicted_behavior is (
        AnswerBehavior.DIRECT_ANSWER
    )

def test_routing_prediction_accepts_none_sufficiency():
    prediction = RoutingPrediction(
        question_id="RQ031",
        predicted_intent=(
            QuestionIntent.AMBIGUOUS
        ),
        predicted_sufficiency=None,
        predicted_behavior=(
            AnswerBehavior.CLARIFY
        ),
    )

    assert (
        prediction.predicted_sufficiency
        is None
    )

def test_routing_prediction_rejects_empty_question_id():
    with pytest.raises(
        ValueError,
        match=(
            "Routing prediction question ID "
            "cannot be empty."
        ),
    ):
        RoutingPrediction(
            question_id=" ",
            predicted_intent=(
                QuestionIntent.FOCUSED
            ),
            predicted_sufficiency=(
                EvidenceSufficiency.SUFFICIENT
            ),
            predicted_behavior=(
                AnswerBehavior.DIRECT_ANSWER
            ),
        )

def test_routing_prediction_rejects_invalid_intent():
    invalid_intent = cast(
        QuestionIntent,
        "focused",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Routing prediction intent must "
            "be a QuestionIntent."
        ),
    ):
        RoutingPrediction(
            question_id="RQ001",
            predicted_intent=invalid_intent,
            predicted_sufficiency=(
                EvidenceSufficiency.SUFFICIENT
            ),
            predicted_behavior=(
                AnswerBehavior.DIRECT_ANSWER
            ),
        )

def test_routing_prediction_rejects_invalid_sufficiency():
    invalid_sufficiency = cast(
        EvidenceSufficiency,
        "sufficient",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Routing prediction sufficiency "
            "must be an EvidenceSufficiency "
            "or None."
        ),
    ):
        RoutingPrediction(
            question_id="RQ001",
            predicted_intent=(
                QuestionIntent.FOCUSED
            ),
            predicted_sufficiency=(
                invalid_sufficiency
            ),
            predicted_behavior=(
                AnswerBehavior.DIRECT_ANSWER
            ),
        )

def test_routing_prediction_rejects_invalid_behavior():
    invalid_behavior = cast(
        AnswerBehavior,
        "direct_answer",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Routing prediction behavior must "
            "be an AnswerBehavior."
        ),
    ):
        RoutingPrediction(
            question_id="RQ001",
            predicted_intent=(
                QuestionIntent.FOCUSED
            ),
            predicted_sufficiency=(
                EvidenceSufficiency.SUFFICIENT
            ),
            predicted_behavior=invalid_behavior,
        )

def test_question_result_accepts_incorrect_prediction():
    result = RoutingQuestionEvaluationResult(
        question_id="RQ001",
        expected_intent=QuestionIntent.FOCUSED,
        predicted_intent=(
            QuestionIntent.AMBIGUOUS
        ),
        expected_sufficiency=(
            EvidenceSufficiency.SUFFICIENT
        ),
        predicted_sufficiency=None,
        expected_behavior=(
            AnswerBehavior.DIRECT_ANSWER
        ),
        predicted_behavior=(
            AnswerBehavior.CLARIFY
        ),
    )

    assert result.predicted_intent is (
        QuestionIntent.AMBIGUOUS
    )

    assert (
        result.predicted_sufficiency
        is None
    )

    assert result.predicted_behavior is (
        AnswerBehavior.CLARIFY
    )

def test_question_result_rejects_invalid_expected_intent():
    invalid_intent = cast(
        QuestionIntent,
        "focused",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Expected routing intent must be "
            "a QuestionIntent."
        ),
    ):
        RoutingQuestionEvaluationResult(
            question_id="RQ001",
            expected_intent=invalid_intent,
            predicted_intent=(
                QuestionIntent.FOCUSED
            ),
            expected_sufficiency=(
                EvidenceSufficiency.SUFFICIENT
            ),
            predicted_sufficiency=(
                EvidenceSufficiency.SUFFICIENT
            ),
            expected_behavior=(
                AnswerBehavior.DIRECT_ANSWER
            ),
            predicted_behavior=(
                AnswerBehavior.DIRECT_ANSWER
            ),
        )

def test_question_result_rejects_invalid_sufficiency():
    invalid_sufficiency = cast(
        EvidenceSufficiency,
        "sufficient",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Routing question result "
            "sufficiency values must be "
            "EvidenceSufficiency or None."
        ),
    ):
        RoutingQuestionEvaluationResult(
            question_id="RQ001",
            expected_intent=(
                QuestionIntent.FOCUSED
            ),
            predicted_intent=(
                QuestionIntent.FOCUSED
            ),
            expected_sufficiency=(
                invalid_sufficiency
            ),
            predicted_sufficiency=(
                EvidenceSufficiency.SUFFICIENT
            ),
            expected_behavior=(
                AnswerBehavior.DIRECT_ANSWER
            ),
            predicted_behavior=(
                AnswerBehavior.DIRECT_ANSWER
            ),
        )

def test_question_result_rejects_invalid_predicted_behavior():
    invalid_behavior = cast(
        AnswerBehavior,
        "direct_answer",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Predicted routing behavior must "
            "be an AnswerBehavior."
        ),
    ):
        RoutingQuestionEvaluationResult(
            question_id="RQ001",
            expected_intent=(
                QuestionIntent.FOCUSED
            ),
            predicted_intent=(
                QuestionIntent.FOCUSED
            ),
            expected_sufficiency=(
                EvidenceSufficiency.SUFFICIENT
            ),
            predicted_sufficiency=(
                EvidenceSufficiency.SUFFICIENT
            ),
            expected_behavior=(
                AnswerBehavior.DIRECT_ANSWER
            ),
            predicted_behavior=invalid_behavior,
        )

def test_routing_class_result_calculates_accuracy():
    result = RoutingClassResult(
        label="focused",
        support=4,
        correct=3,
    )

    assert result.accuracy == 0.75

def test_routing_class_result_rejects_nonpositive_support():
    with pytest.raises(
        ValueError,
        match=(
            "Routing class result support "
            "must be greater than zero."
        ),
    ):
        RoutingClassResult(
            label="focused",
            support=0,
            correct=0,
        )

def test_routing_class_result_rejects_negative_correct_count():
    with pytest.raises(
        ValueError,
        match=(
            "Routing class result correct "
            "count cannot be negative."
        ),
    ):
        RoutingClassResult(
            label="focused",
            support=1,
            correct=-1,
        )

def test_routing_class_result_rejects_correct_above_support():
    with pytest.raises(
        ValueError,
        match=(
            "Routing class result correct "
            "count cannot exceed support."
        ),
    ):
        RoutingClassResult(
            label="focused",
            support=1,
            correct=2,
        )

def test_classification_result_passes_at_macro_threshold():
    result = RoutingClassificationResult(
        overall_accuracy=0.95,
        macro_accuracy=0.95,
        class_results=(
            RoutingClassResult(
                label="focused",
                support=20,
                correct=19,
            ),
        ),
        pass_threshold=0.95,
    )

    assert result.passed

def test_classification_result_uses_macro_for_quality_gate():
    result = RoutingClassificationResult(
        overall_accuracy=0.99,
        macro_accuracy=0.94,
        class_results=(
            RoutingClassResult(
                label="focused",
                support=100,
                correct=94,
            ),
        ),
        pass_threshold=0.95,
    )

    assert not result.passed

def test_classification_result_rejects_empty_class_results():
    with pytest.raises(
        ValueError,
        match=(
            "Routing classification result "
            "must contain class results."
        ),
    ):
        RoutingClassificationResult(
            overall_accuracy=1.0,
            macro_accuracy=1.0,
            class_results=(),
            pass_threshold=0.95,
        )

def test_classification_result_rejects_duplicate_labels():
    with pytest.raises(
        ValueError,
        match=(
            "Routing classification class "
            "labels must be unique."
        ),
    ):
        RoutingClassificationResult(
            overall_accuracy=1.0,
            macro_accuracy=1.0,
            class_results=(
                RoutingClassResult(
                    label="focused",
                    support=1,
                    correct=1,
                ),
                RoutingClassResult(
                    label="focused",
                    support=1,
                    correct=1,
                ),
            ),
            pass_threshold=0.95,
        )

def test_classification_result_rejects_invalid_ranges():
    class_results = (
        RoutingClassResult(
            label="focused",
            support=1,
            correct=1,
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Routing overall accuracy must "
            "be between 0 and 1."
        ),
    ):
        RoutingClassificationResult(
            overall_accuracy=1.01,
            macro_accuracy=1.0,
            class_results=class_results,
            pass_threshold=0.95,
        )

    with pytest.raises(
        ValueError,
        match=(
            "Routing macro accuracy must "
            "be between 0 and 1."
        ),
    ):
        RoutingClassificationResult(
            overall_accuracy=1.0,
            macro_accuracy=-0.01,
            class_results=class_results,
            pass_threshold=0.95,
        )

    with pytest.raises(
        ValueError,
        match=(
            "Routing classification pass "
            "threshold must be between "
            "0 and 1."
        ),
    ):
        RoutingClassificationResult(
            overall_accuracy=1.0,
            macro_accuracy=1.0,
            class_results=class_results,
            pass_threshold=1.01,
        )

def test_evaluation_run_result_passes_when_all_layers_pass():
    passing = _make_classification_result()

    result = RoutingEvaluationRunResult(
        question_results=(
            _make_question_result(),
        ),
        intent_result=passing,
        sufficiency_result=passing,
        behavior_result=passing,
    )

    assert result.passed

def test_evaluation_run_result_fails_when_one_layer_fails():
    passing = _make_classification_result()

    failing = _make_classification_result(
        accuracy=0.94,
    )

    result = RoutingEvaluationRunResult(
        question_results=(
            _make_question_result(),
        ),
        intent_result=passing,
        sufficiency_result=failing,
        behavior_result=passing,
    )

    assert not result.passed

def test_evaluation_run_result_rejects_empty_question_results():
    passing = _make_classification_result()

    with pytest.raises(
        ValueError,
        match=(
            "Routing evaluation run must "
            "contain question results."
        ),
    ):
        RoutingEvaluationRunResult(
            question_results=(),
            intent_result=passing,
            sufficiency_result=passing,
            behavior_result=passing,
        )

def test_evaluation_run_result_rejects_duplicate_question_ids():
    passing = _make_classification_result()

    with pytest.raises(
        ValueError,
        match=(
            "Routing evaluation question "
            "result IDs must be unique."
        ),
    ):
        RoutingEvaluationRunResult(
            question_results=(
                _make_question_result(
                    "RQ001"
                ),
                _make_question_result(
                    "RQ001"
                ),
            ),
            intent_result=passing,
            sufficiency_result=passing,
            behavior_result=passing,
        )

def test_evaluation_run_result_rejects_invalid_question_result():
    passing = _make_classification_result()

    invalid_results = cast(
        tuple[
            RoutingQuestionEvaluationResult,
            ...
        ],
        ("invalid",),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Routing evaluation question "
            "results must be "
            "RoutingQuestionEvaluationResult "
            "values."
        ),
    ):
        RoutingEvaluationRunResult(
            question_results=invalid_results,
            intent_result=passing,
            sufficiency_result=passing,
            behavior_result=passing,
        )

def test_evaluation_run_result_rejects_invalid_classification_result():
    passing = _make_classification_result()

    invalid_result = cast(
        RoutingClassificationResult,
        "invalid",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Routing evaluation classification "
            "results must be "
            "RoutingClassificationResult "
            "values."
        ),
    ):
        RoutingEvaluationRunResult(
            question_results=(
                _make_question_result(),
            ),
            intent_result=passing,
            sufficiency_result=(
                invalid_result
            ),
            behavior_result=passing,
        )