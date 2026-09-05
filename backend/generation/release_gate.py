"""
Enforce final grounding checks before generated answers are released.

This module combines deterministic citation validation with
generated-claim grounding validation. Generated output is returned only
when all required release checks pass.

Failures are represented explicitly and allow the application service
to fail closed rather than expose an answer whose evidence support
cannot be established.
"""

from dataclasses import dataclass
from backend.generation.citation_validator import (
    validate_generation_citations,
)
from backend.generation.claim_grounding_validator import (
    GeneratedClaimGroundingValidator,
)
from backend.generation.grounded_generator import (
    GroundedAnswerGenerator,
    GroundedGenerationRequest,
    generate_grounded_answer,
)

class GroundedGenerationRejectedError(
    RuntimeError
):
    """Signal that generated output failed a mandatory grounding release check.
    """

    pass

@dataclass(frozen=True)
class ReleasedGroundedAnswer:
    """Represent generated answer text that passed all required release checks.
    """

    text: str
    cited_evidence_indexes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError(
                "Released grounded answer "
                "cannot be empty."
            )

        if not self.cited_evidence_indexes:
            raise ValueError(
                "Released grounded answer must "
                "identify cited evidence."
            )

        if any(
            index < 1
            for index
            in self.cited_evidence_indexes
        ):
            raise ValueError(
                "Released evidence indexes must "
                "be positive."
            )

def generate_validated_grounded_answer(
    request: GroundedGenerationRequest,
    generator: GroundedAnswerGenerator,
    claim_grounding_validator: (
        GeneratedClaimGroundingValidator
    ),
) -> ReleasedGroundedAnswer:
    """Generate an answer and enforce citation and claim-grounding validation.

    Any failed validation raises GroundedGenerationRejectedError so
    unsupported generated content is not released as grounded output.
    """
    generation_result = (
        generate_grounded_answer(
            request,
            generator,
        )
    )

    citation_validation = (
        validate_generation_citations(
            request,
            generation_result,
        )
    )

    if not citation_validation.valid:
        raise GroundedGenerationRejectedError(
            citation_validation.reason
        )

    grounding_validation = (
        claim_grounding_validator.validate(
            request,
            generation_result,
        )
    )

    if not grounding_validation.valid:
        raise GroundedGenerationRejectedError(
            grounding_validation.reason
        )

    return ReleasedGroundedAnswer(
        text=generation_result.text,
        cited_evidence_indexes=(
            citation_validation
            .cited_evidence_indexes
        ),
    )