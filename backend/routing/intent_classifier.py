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