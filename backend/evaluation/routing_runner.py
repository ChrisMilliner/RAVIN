from enum import Enum
from typing import Callable
from backend.behavior import AnswerBehavior
from backend.evaluation.routing_metrics import (
    calculate_class_accuracy,
    calculate_classification_accuracy,
    calculate_macro_accuracy,
)
from backend.evaluation.routing_models import (
    RoutingClassResult,
    RoutingClassificationResult,
    RoutingEvaluationConfig,
    RoutingEvaluationQuestion,
    RoutingEvaluationRunResult,
    RoutingPrediction,
    RoutingQuestionEvaluationResult,
)
from backend.routing.models import (
    EvidenceSufficiency,
    QuestionIntent,
)

RoutingPredictionFunction = Callable[
    [RoutingEvaluationQuestion],
    RoutingPrediction,
]

def _build_class_result(
    expected_labels: tuple[Enum, ...],
    predicted_labels: tuple[Enum, ...],
    label: Enum,
) -> RoutingClassResult:
    support = sum(
        1
        for expected in expected_labels
        if expected == label
    )

    accuracy = calculate_class_accuracy(
        expected_labels,
        predicted_labels,
        label,
    )

    correct = sum(
        1
        for expected, predicted in zip(
            expected_labels,
            predicted_labels,
        )
        if (
            expected == label
            and predicted == label
        )
    )

    if support <= 0:
        raise RuntimeError(
            "Routing evaluation class support "
            "must be greater than zero."
        )

    if correct / support != accuracy:
        raise RuntimeError(
            "Routing class accuracy calculation "
            "is inconsistent."
        )

    return RoutingClassResult(
        label=str(label.value),
        support=support,
        correct=correct,
    )

def _build_classification_result(
    expected_labels: tuple[Enum, ...],
    predicted_labels: tuple[Enum, ...],
    labels: tuple[Enum, ...],
    pass_threshold: float,
) -> RoutingClassificationResult:
    overall_accuracy = (
        calculate_classification_accuracy(
            expected_labels,
            predicted_labels,
        )
    )

    macro_accuracy = calculate_macro_accuracy(
        expected_labels,
        predicted_labels,
        labels,
    )

    class_results = tuple(
        _build_class_result(
            expected_labels,
            predicted_labels,
            label,
        )
        for label in labels
    )

    return RoutingClassificationResult(
        overall_accuracy=overall_accuracy,
        macro_accuracy=macro_accuracy,
        class_results=class_results,
        pass_threshold=pass_threshold,
    )

def run_routing_evaluation(
    questions: tuple[
        RoutingEvaluationQuestion,
        ...
    ],
    predict: RoutingPredictionFunction,
    config: RoutingEvaluationConfig,
) -> RoutingEvaluationRunResult:
    if not questions:
        raise ValueError(
            "Cannot evaluate an empty routing "
            "question set."
        )

    question_results: list[
        RoutingQuestionEvaluationResult
    ] = []

    predictions: list[
        RoutingPrediction
    ] = []

    for question in questions:
        prediction = predict(question)

        if not isinstance(
            prediction,
            RoutingPrediction,
        ):
            raise ValueError(
                "Routing prediction function must "
                "return a RoutingPrediction."
            )

        if (
            prediction.question_id
            != question.question_id
        ):
            raise ValueError(
                "Routing prediction question ID "
                "must match the evaluation "
                "question ID."
            )

        predictions.append(
            prediction
        )

        question_results.append(
            RoutingQuestionEvaluationResult(
                question_id=question.question_id,
                expected_intent=(
                    question.expected_intent
                ),
                predicted_intent=(
                    prediction.predicted_intent
                ),
                expected_sufficiency=(
                    question.expected_sufficiency
                ),
                predicted_sufficiency=(
                    prediction.predicted_sufficiency
                ),
                expected_behavior=(
                    question.expected_behavior
                ),
                predicted_behavior=(
                    prediction.predicted_behavior
                ),
            )
        )

    expected_intents = tuple(
        question.expected_intent
        for question in questions
    )

    predicted_intents = tuple(
        prediction.predicted_intent
        for prediction in predictions
    )

    intent_result = (
        _build_classification_result(
            expected_intents,
            predicted_intents,
            (
                QuestionIntent.FOCUSED,
                QuestionIntent.BROAD,
                QuestionIntent.AMBIGUOUS,
            ),
            config.intent_pass_threshold,
        )
    )

    clear_pairs = tuple(
        (
            question,
            prediction,
        )
        for question, prediction in zip(
            questions,
            predictions,
        )
        if (
            question.expected_sufficiency
            is not None
        )
    )

    if not clear_pairs:
        raise ValueError(
            "Routing sufficiency evaluation "
            "requires at least one clear "
            "question."
        )

    expected_sufficiency = tuple(
        question.expected_sufficiency
        for question, _ in clear_pairs
    )

    if any(
        sufficiency is None
        for sufficiency in expected_sufficiency
    ):
        raise RuntimeError(
            "Clear routing questions must "
            "define expected sufficiency."
        )

    expected_sufficiency_labels = tuple(
        sufficiency
        for sufficiency in expected_sufficiency
        if sufficiency is not None
    )

    predicted_sufficiency_labels = tuple(
        (
            prediction.predicted_sufficiency
            if (
                prediction.predicted_sufficiency
                is not None
            )
            else EvidenceSufficiency.UNCERTAIN
        )
        for _, prediction in clear_pairs
    )

    sufficiency_result = (
        _build_classification_result(
            expected_sufficiency_labels,
            predicted_sufficiency_labels,
            (
                EvidenceSufficiency.SUFFICIENT,
                EvidenceSufficiency.INSUFFICIENT,
            ),
            config.sufficiency_pass_threshold,
        )
    )

    expected_behaviors = tuple(
        question.expected_behavior
        for question in questions
    )

    predicted_behaviors = tuple(
        prediction.predicted_behavior
        for prediction in predictions
    )

    behavior_result = (
        _build_classification_result(
            expected_behaviors,
            predicted_behaviors,
            (
                AnswerBehavior.DIRECT_ANSWER,
                AnswerBehavior.GROUNDED_OVERVIEW,
                AnswerBehavior.CLARIFY,
                AnswerBehavior.NO_GROUNDED_ANSWER,
            ),
            config.behavior_pass_threshold,
        )
    )

    return RoutingEvaluationRunResult(
        question_results=tuple(
            question_results
        ),
        intent_result=intent_result,
        sufficiency_result=(
            sufficiency_result
        ),
        behavior_result=behavior_result,
    )