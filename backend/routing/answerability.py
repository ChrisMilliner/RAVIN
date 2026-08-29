"""
Define the framework-neutral answerability scoring contract.

Answerability scoring estimates how strongly retrieved policy context
addresses a material proposition extracted from the user's question.
The resulting scores support proposition-level evidence coverage
assessment.

Answerability scores are evidence signals rather than direct answer
behaviour decisions and are not themselves accuracy measurements.
"""

from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class AnswerabilityResult:
    scores: tuple[
        float,
        ...
    ]

    def __post_init__(self) -> None:
        if not self.scores:
            raise ValueError(
                "Answerability scores cannot "
                "be empty."
            )

        if any(
            not 0.0 <= score <= 1.0
            for score in self.scores
        ):
            raise ValueError(
                "Answerability scores must be "
                "between 0 and 1."
            )

    @property
    def strongest_score(
        self,
    ) -> float:
        return max(
            self.scores
        )

class AnswerabilityProvider(
    Protocol
):
    def score(
        self,
        question: str,
        evidence_texts: tuple[
            str,
            ...
        ],
    ) -> AnswerabilityResult:
        ...

def score_answerability(
    question: str,
    evidence_texts: tuple[
        str,
        ...
    ],
    provider: AnswerabilityProvider,
) -> AnswerabilityResult:
    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    if not evidence_texts:
        raise ValueError(
            "Answerability evidence cannot "
            "be empty."
        )

    if any(
        not evidence_text.strip()
        for evidence_text
        in evidence_texts
    ):
        raise ValueError(
            "Answerability evidence cannot "
            "contain empty text."
        )

    result = provider.score(
        question,
        evidence_texts,
    )

    if not isinstance(
        result,
        AnswerabilityResult,
    ):
        raise ValueError(
            "Answerability provider must return "
            "an AnswerabilityResult."
        )

    if (
        len(result.scores)
        != len(evidence_texts)
    ):
        raise ValueError(
            "Answerability provider must return "
            "one score per evidence text."
        )

    return result