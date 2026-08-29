"""
Classify question intent using deterministic RAVIN rules.

This implementation distinguishes focused, broad, and ambiguous
questions using explicit structural signals rather than generative
model judgement.

The classifier exists to keep answer-control behaviour explainable,
repeatable, and testable independently of the language model used for
grounded wording.
"""

import re
from backend.routing.models import (
    QuestionIntent,
)

_TOPIC_ANCHORS = (
    "academic dress",
    "academic progress",
    "academic progression",
    "academic promotion",
    "promotion",
    "professional equivalence",
    "admission",
    "admissions",
    "applicant",
    "student",
    "staff",
    "graduate",
    "course",
    "offer",
    "single subject",
    "special entry",
    "seas",
    "referee",
    "assessor",
    "qualification",
    "enrolment",
    "enrollment",
    "academic hood",
    "stole",
    "academic work",
    "show cause",
    "defer",
    "deferral",
    "transfer",
)

def _normalize_question(
    question: str,
) -> str:
    return " ".join(
        question.strip().lower().split()
    )

def _contains_topic_anchor(
    question: str,
) -> bool:
    return any(
        anchor in question
        for anchor in _TOPIC_ANCHORS
    )

def _contains_multi_part_request(
    question: str,
) -> bool:
    comma_count = question.count(",")

    has_and = bool(
        re.search(
            r"\band\b",
            question,
        )
    )

    asks_for_categories = (
        question.startswith("what ")
        or question.startswith("which ")
    )

    return (
        asks_for_categories
        and comma_count >= 1
        and has_and
    )

def _contains_broad_signal(
    question: str,
) -> bool:
    if _contains_multi_part_request(
        question
    ):
        return True

    broad_phrases = (
        "at each stage",
        "what happens when",
        "how does ",
        "what are the admission requirements",
        "what are the admissions requirements",
        "what admissions information",
    )

    return any(
        phrase in question
        for phrase in broad_phrases
    )

class RuleBasedQuestionIntentClassifier:
    def classify(
        self,
        question: str,
    ) -> QuestionIntent:
        normalized = _normalize_question(
            question
        )

        if not normalized:
            raise ValueError(
                "Question cannot be empty."
            )

        if not _contains_topic_anchor(
            normalized
        ):
            return QuestionIntent.AMBIGUOUS

        if _contains_broad_signal(
            normalized
        ):
            return QuestionIntent.BROAD

        return QuestionIntent.FOCUSED