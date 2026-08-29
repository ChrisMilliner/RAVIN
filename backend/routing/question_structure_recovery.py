"""
Define the contract for deterministic question-structure recovery.

Recovery providers receive question parses that could not be accepted
with sufficient confidence and may return a supported corrected
structure for known patterns.

Recovery is bounded and deterministic. It is not a general language
model fallback and must fail unresolved when no supported recovery rule
applies.
"""

from typing import Protocol
from backend.routing.question_parser import (
    QuestionParse,
    QuestionParseResult,
)

class QuestionStructureRecoveryProvider(
    Protocol
):
    def recover(
        self,
        question: str,
        parse_result: QuestionParseResult,
    ) -> QuestionParse | None:
        ...

def recover_question_structure(
    question: str,
    parse_result: QuestionParseResult,
    provider: QuestionStructureRecoveryProvider,
) -> QuestionParse | None:
    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    if not isinstance(
        parse_result,
        QuestionParseResult,
    ):
        raise ValueError(
            "Parse result must be a "
            "QuestionParseResult."
        )

    recovered = provider.recover(
        question,
        parse_result,
    )

    if (
        recovered is not None
        and not isinstance(
            recovered,
            QuestionParse,
        )
    ):
        raise ValueError(
            "Question structure recovery "
            "provider must return a "
            "QuestionParse or None."
        )

    return recovered