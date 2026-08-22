from dataclasses import dataclass
from backend.behavior import AnswerBehavior
from backend.routing.models import (
    EvidenceSufficiency,
    QuestionIntent,
)

@dataclass(frozen=True)
class RoutingEvaluationQuestion:
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