"""
Define the framework-neutral question-intent classification contract.

Question intent describes whether a request is focused, broad, or
ambiguous before final answer behaviour is selected. The classifier
contract allows deterministic implementations to be used without
coupling routing orchestration to their internal rules.

Intent classification is a control decision and is not delegated to a
generative language model.
"""

from typing import Protocol
from backend.routing.models import (
    QuestionIntent,
)

class QuestionIntentClassifier(Protocol):
    def classify(
        self,
        question: str,
    ) -> QuestionIntent:
        ...

def classify_question_intent(
    question: str,
    classifier: QuestionIntentClassifier,
) -> QuestionIntent:
    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    intent = classifier.classify(
        question
    )

    if not isinstance(
        intent,
        QuestionIntent,
    ):
        raise ValueError(
            "Question intent classifier returned "
            "an invalid intent."
        )

    return intent