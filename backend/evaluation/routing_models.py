"""
Define models used by RAVIN routing evaluation.

These models represent routing questions, expected classifications,
predictions, per-class results, aggregate results, and configured pass
thresholds for intent, evidence sufficiency, and answer behaviour.

The default development threshold does not constitute a validated
accuracy result by itself.
"""

from dataclasses import dataclass
from backend.behavior import AnswerBehavior
from backend.routing.models import (
    EvidenceSufficiency,
    QuestionIntent,
)

@dataclass(frozen=True)
class RoutingEvaluationQuestion:
    """Represent expected intent, sufficiency, and behavior for one routing question.
    """

    question_id: str
    question: str
    expected_intent: QuestionIntent
    expected_sufficiency: (
        EvidenceSufficiency | None
    )
    expected_behavior: AnswerBehavior
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError(
                "Routing evaluation question ID "
                "cannot be empty."
            )

        if not self.question.strip():
            raise ValueError(
                "Routing evaluation question "
                "cannot be empty."
            )

        if not isinstance(
            self.expected_intent,
            QuestionIntent,
        ):
            raise ValueError(
                "Expected intent must be a "
                "QuestionIntent."
            )

        if (
            self.expected_sufficiency is not None
            and not isinstance(
                self.expected_sufficiency,
                EvidenceSufficiency,
            )
        ):
            raise ValueError(
                "Expected sufficiency must be an "
                "EvidenceSufficiency or None."
            )

        if not isinstance(
            self.expected_behavior,
            AnswerBehavior,
        ):
            raise ValueError(
                "Expected behavior must be an "
                "AnswerBehavior."
            )

        if (
            self.notes is not None
            and not isinstance(
                self.notes,
                str,
            )
        ):
            raise ValueError(
                "Routing evaluation notes must "
                "be a string or None."
            )

        self._validate_expected_route()

    def _validate_expected_route(
        self,
    ) -> None:
        if (
            self.expected_intent
            == QuestionIntent.AMBIGUOUS
        ):
            if (
                self.expected_sufficiency
                is not None
            ):
                raise ValueError(
                    "Ambiguous routing questions "
                    "must not define expected "
                    "evidence sufficiency."
                )

            if (
                self.expected_behavior
                != AnswerBehavior.CLARIFY
            ):
                raise ValueError(
                    "Ambiguous routing questions "
                    "must expect clarify behavior."
                )

            return

        if self.expected_sufficiency is None:
            raise ValueError(
                "Clear routing questions must "
                "define expected evidence "
                "sufficiency."
            )

        if (
            self.expected_sufficiency
            == EvidenceSufficiency.UNCERTAIN
        ):
            raise ValueError(
                "Uncertain is a runtime fallback "
                "and cannot be evaluation truth."
            )

        if (
            self.expected_sufficiency
            == EvidenceSufficiency.INSUFFICIENT
        ):
            if (
                self.expected_behavior
                != AnswerBehavior.NO_GROUNDED_ANSWER
            ):
                raise ValueError(
                    "Insufficient evidence must "
                    "expect no-grounded-answer "
                    "behavior."
                )

            return

        if (
            self.expected_intent
            == QuestionIntent.FOCUSED
        ):
            expected_behavior = (
                AnswerBehavior.DIRECT_ANSWER
            )
        else:
            expected_behavior = (
                AnswerBehavior.GROUNDED_OVERVIEW
            )

        if (
            self.expected_behavior
            != expected_behavior
        ):
            raise ValueError(
                "Expected behavior does not match "
                "the intent and evidence labels."
            )

DEFAULT_ROUTING_PASS_THRESHOLD = 0.95

@dataclass(frozen=True)
class RoutingEvaluationConfig:
    """Configure pass thresholds for routing intent, sufficiency, and behavior.
    """

    intent_pass_threshold: float = (
        DEFAULT_ROUTING_PASS_THRESHOLD
    )
    sufficiency_pass_threshold: float = (
        DEFAULT_ROUTING_PASS_THRESHOLD
    )
    behavior_pass_threshold: float = (
        DEFAULT_ROUTING_PASS_THRESHOLD
    )

    def __post_init__(self) -> None:
        thresholds = (
            self.intent_pass_threshold,
            self.sufficiency_pass_threshold,
            self.behavior_pass_threshold,
        )

        if any(
            not 0.0 <= threshold <= 1.0
            for threshold in thresholds
        ):
            raise ValueError(
                "Routing evaluation thresholds "
                "must be between 0 and 1."
            )

@dataclass(frozen=True)
class RoutingPrediction:
    """Represent routing predictions produced for one evaluation question.
    """

    question_id: str
    predicted_intent: QuestionIntent
    predicted_sufficiency: (
        EvidenceSufficiency | None
    )
    predicted_behavior: AnswerBehavior

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError(
                "Routing prediction question ID "
                "cannot be empty."
            )

        if not isinstance(
            self.predicted_intent,
            QuestionIntent,
        ):
            raise ValueError(
                "Routing prediction intent must "
                "be a QuestionIntent."
            )

        if (
            self.predicted_sufficiency is not None
            and not isinstance(
                self.predicted_sufficiency,
                EvidenceSufficiency,
            )
        ):
            raise ValueError(
                "Routing prediction sufficiency "
                "must be an EvidenceSufficiency "
                "or None."
            )

        if not isinstance(
            self.predicted_behavior,
            AnswerBehavior,
        ):
            raise ValueError(
                "Routing prediction behavior must "
                "be an AnswerBehavior."
            )

@dataclass(frozen=True)
class RoutingQuestionEvaluationResult:
    """Record expected and predicted routing states for one question.
    """

    question_id: str
    expected_intent: QuestionIntent
    predicted_intent: QuestionIntent
    expected_sufficiency: (
        EvidenceSufficiency | None
    )
    predicted_sufficiency: (
        EvidenceSufficiency | None
    )
    expected_behavior: AnswerBehavior
    predicted_behavior: AnswerBehavior

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError(
                "Routing question result ID "
                "cannot be empty."
            )

        if not isinstance(
            self.expected_intent,
            QuestionIntent,
        ):
            raise ValueError(
                "Expected routing intent must be "
                "a QuestionIntent."
            )

        if not isinstance(
            self.predicted_intent,
            QuestionIntent,
        ):
            raise ValueError(
                "Predicted routing intent must be "
                "a QuestionIntent."
            )

        for sufficiency in (
            self.expected_sufficiency,
            self.predicted_sufficiency,
        ):
            if (
                sufficiency is not None
                and not isinstance(
                    sufficiency,
                    EvidenceSufficiency,
                )
            ):
                raise ValueError(
                    "Routing question result "
                    "sufficiency values must be "
                    "EvidenceSufficiency or None."
                )

        if not isinstance(
            self.expected_behavior,
            AnswerBehavior,
        ):
            raise ValueError(
                "Expected routing behavior must "
                "be an AnswerBehavior."
            )

        if not isinstance(
            self.predicted_behavior,
            AnswerBehavior,
        ):
            raise ValueError(
                "Predicted routing behavior must "
                "be an AnswerBehavior."
            )

@dataclass(frozen=True)
class RoutingClassResult:
    """Record support and correct predictions for one routing class.
    """

    label: str
    support: int
    correct: int

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError(
                "Routing class result label "
                "cannot be empty."
            )

        if self.support <= 0:
            raise ValueError(
                "Routing class result support "
                "must be greater than zero."
            )

        if self.correct < 0:
            raise ValueError(
                "Routing class result correct "
                "count cannot be negative."
            )

        if self.correct > self.support:
            raise ValueError(
                "Routing class result correct "
                "count cannot exceed support."
            )

    @property
    def accuracy(self) -> float:
        """Return accuracy for this routing class.
        """
        return self.correct / self.support

@dataclass(frozen=True)
class RoutingClassificationResult:
    """Aggregate overall, macro, and per-class routing classification results.
    """

    overall_accuracy: float
    macro_accuracy: float
    class_results: tuple[
        RoutingClassResult,
        ...
    ]
    pass_threshold: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.overall_accuracy <= 1.0:
            raise ValueError(
                "Routing overall accuracy must "
                "be between 0 and 1."
            )

        if not 0.0 <= self.macro_accuracy <= 1.0:
            raise ValueError(
                "Routing macro accuracy must "
                "be between 0 and 1."
            )

        if not self.class_results:
            raise ValueError(
                "Routing classification result "
                "must contain class results."
            )

        if any(
            not isinstance(
                result,
                RoutingClassResult,
            )
            for result in self.class_results
        ):
            raise ValueError(
                "Routing classification class "
                "results must be "
                "RoutingClassResult values."
            )

        labels = tuple(
            result.label
            for result in self.class_results
        )

        if len(set(labels)) != len(labels):
            raise ValueError(
                "Routing classification class "
                "labels must be unique."
            )

        if not 0.0 <= self.pass_threshold <= 1.0:
            raise ValueError(
                "Routing classification pass "
                "threshold must be between "
                "0 and 1."
            )

    @property
    def passed(self) -> bool:
        """Return whether macro accuracy meets the configured pass threshold.
        """
        return (
            self.macro_accuracy
            >= self.pass_threshold
        )

@dataclass(frozen=True)
class RoutingEvaluationRunResult:
    """Aggregate intent, sufficiency, behavior, and per-question routing results.
    """

    question_results: tuple[
        RoutingQuestionEvaluationResult,
        ...
    ]
    intent_result: RoutingClassificationResult
    sufficiency_result: RoutingClassificationResult
    behavior_result: RoutingClassificationResult

    def __post_init__(self) -> None:
        if not self.question_results:
            raise ValueError(
                "Routing evaluation run must "
                "contain question results."
            )

        if any(
            not isinstance(
                result,
                RoutingQuestionEvaluationResult,
            )
            for result in self.question_results
        ):
            raise ValueError(
                "Routing evaluation question "
                "results must be "
                "RoutingQuestionEvaluationResult "
                "values."
            )

        question_ids = tuple(
            result.question_id
            for result in self.question_results
        )

        if len(set(question_ids)) != len(
            question_ids
        ):
            raise ValueError(
                "Routing evaluation question "
                "result IDs must be unique."
            )

        classification_results = (
            self.intent_result,
            self.sufficiency_result,
            self.behavior_result,
        )

        if any(
            not isinstance(
                result,
                RoutingClassificationResult,
            )
            for result in classification_results
        ):
            raise ValueError(
                "Routing evaluation classification "
                "results must be "
                "RoutingClassificationResult "
                "values."
            )

    @property
    def passed(self) -> bool:
        """Return whether all three routing evaluation dimensions passed.
        """
        return (
            self.intent_result.passed
            and self.sufficiency_result.passed
            and self.behavior_result.passed
        )