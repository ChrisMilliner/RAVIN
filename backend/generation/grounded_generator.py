"""
Define framework-neutral contracts for grounded answer generation.

This module contains the request, result, and generator interfaces used
after RAVIN has already classified the question, retrieved policy
evidence, assessed evidence sufficiency, and selected an answer
behaviour.

Generation is therefore a wording stage rather than a control stage.
Implementations receive approved evidence and must not independently
decide routing or evidence sufficiency.
"""

from dataclasses import dataclass
from typing import Protocol
from backend.behavior import AnswerBehavior

_ALLOWED_GENERATION_BEHAVIORS = {
    AnswerBehavior.DIRECT_ANSWER,
    AnswerBehavior.GROUNDED_OVERVIEW,
}

@dataclass(frozen=True)
class GroundedGenerationRequest:
    question: str
    behavior: AnswerBehavior
    evidence_texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError(
                "Generation question cannot be empty."
            )

        if self.behavior not in (
            _ALLOWED_GENERATION_BEHAVIORS
        ):
            raise ValueError(
                "Grounded generation is only "
                "allowed for answer-producing "
                "behaviors."
            )

        if not self.evidence_texts:
            raise ValueError(
                "Grounded generation requires "
                "evidence."
            )

        if any(
            not text.strip()
            for text in self.evidence_texts
        ):
            raise ValueError(
                "Generation evidence cannot "
                "contain empty text."
            )

@dataclass(frozen=True)
class GroundedGenerationResult:
    text: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError(
                "Generated answer cannot be empty."
            )

class GroundedAnswerGenerator(Protocol):
    def generate(
        self,
        request: GroundedGenerationRequest,
    ) -> GroundedGenerationResult:
        ...

def generate_grounded_answer(
    request: GroundedGenerationRequest,
    generator: GroundedAnswerGenerator,
) -> GroundedGenerationResult:
    if not isinstance(
        request,
        GroundedGenerationRequest,
    ):
        raise ValueError(
            "Generation request must be a "
            "GroundedGenerationRequest."
        )

    result = generator.generate(
        request
    )

    if not isinstance(
        result,
        GroundedGenerationResult,
    ):
        raise ValueError(
            "Grounded answer generator must "
            "return a GroundedGenerationResult."
        )

    return result