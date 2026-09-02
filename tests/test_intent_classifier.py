from typing import cast
import pytest
from backend.routing.intent_classifier import (
    classify_question_intent,
)
from backend.routing.models import (
    QuestionIntent,
)

class FakeQuestionIntentClassifier:
    def __init__(
        self,
        intent: QuestionIntent,
    ) -> None:
        self.intent = intent
        self.question: str | None = None

    def classify(
        self,
        question: str,
    ) -> QuestionIntent:
        self.question = question

        return self.intent

def test_classify_question_intent_returns_focused():
    classifier = FakeQuestionIntentClassifier(
        QuestionIntent.FOCUSED
    )

    intent = classify_question_intent(
        "Who approves academic dress?",
        classifier,
    )

    assert intent is QuestionIntent.FOCUSED

    assert classifier.question == (
        "Who approves academic dress?"
    )

def test_classify_question_intent_returns_broad():
    classifier = FakeQuestionIntentClassifier(
        QuestionIntent.BROAD
    )

    intent = classify_question_intent(
        (
            "What happens when a student is not "
            "making satisfactory progress?"
        ),
        classifier,
    )

    assert intent is QuestionIntent.BROAD

def test_classify_question_intent_returns_ambiguous():
    classifier = FakeQuestionIntentClassifier(
        QuestionIntent.AMBIGUOUS
    )

    intent = classify_question_intent(
        "What do I need to submit?",
        classifier,
    )

    assert intent is QuestionIntent.AMBIGUOUS

def test_classify_question_intent_strips_question():
    classifier = FakeQuestionIntentClassifier(
        QuestionIntent.FOCUSED
    )

    classify_question_intent(
        "  Who approves academic dress?  ",
        classifier,
    )

    assert classifier.question == (
        "Who approves academic dress?"
    )

def test_classify_question_intent_rejects_empty_question():
    classifier = FakeQuestionIntentClassifier(
        QuestionIntent.FOCUSED
    )

    with pytest.raises(
        ValueError,
        match="Question cannot be empty.",
    ):
        classify_question_intent(
            "   ",
            classifier,
        )

def test_classify_question_intent_rejects_invalid_result():
    invalid_intent = cast(
        QuestionIntent,
        "focused",
    )

    classifier = FakeQuestionIntentClassifier(
        invalid_intent
    )

    with pytest.raises(
        ValueError,
        match=(
            "Question intent classifier returned "
            "an invalid intent."
        ),
    ):
        classify_question_intent(
            "Question?",
            classifier,
        )