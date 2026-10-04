"""
Enforce final grounding checks before generated answers are released.

This module combines deterministic citation validation with
generated-claim grounding validation. Fully valid generated output is
released unchanged. When only some generated claims pass grounding
validation, unsupported claims are removed and only validated claims
are released.

Failures are represented explicitly and allow the application service
to fail closed when no generated claims can be safely released.
"""

from dataclasses import dataclass
from backend.generation.citation_validator import (
    validate_generation_citations,
)
from backend.generation.claim_grounding_validator import (
    ClaimGroundingResult,
    GeneratedClaimGroundingValidator,
)
from backend.generation.grounded_generator import (
    GroundedAnswerGenerator,
    GroundedGenerationRequest,
    GroundedGenerationResult,
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
    """Represent generated answer text that passed required release checks.
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

    Fully valid output is released unchanged. If some generated claims fail
    grounding validation, only supported cited claims are retained and the
    filtered answer is citation-validated before release. If no supported
    claims remain, generation is rejected and the caller can fail closed.
    """
    generation_result = (
        generate_grounded_answer(
            request,
            generator,
        )
    )

    grounding_validation = (
        claim_grounding_validator.validate(
            request,
            generation_result,
        )
    )

    if grounding_validation.valid:
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

        return ReleasedGroundedAnswer(
            text=generation_result.text,
            cited_evidence_indexes=(
                citation_validation
                .cited_evidence_indexes
            ),
        )

    supported_claims = tuple(
        claim
        for claim in grounding_validation.claims
        if (
            claim.supported
            and claim.cited_evidence_indexes
        )
    )

    if not supported_claims:
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

        raise GroundedGenerationRejectedError(
            grounding_validation.reason
        )

    filtered_text = " ".join(
        _render_supported_claim(
            claim
        )
        for claim in supported_claims
    )

    filtered_result = GroundedGenerationResult(
        text=filtered_text
    )

    citation_validation = (
        validate_generation_citations(
            request,
            filtered_result,
        )
    )

    if not citation_validation.valid:
        raise GroundedGenerationRejectedError(
            citation_validation.reason
        )

    return ReleasedGroundedAnswer(
        text=filtered_text,
        cited_evidence_indexes=(
            citation_validation
            .cited_evidence_indexes
        ),
    )

def _render_supported_claim(
    claim: ClaimGroundingResult,
) -> str:
    evidence_markers = " ".join(
        f"[E{index}]"
        for index
        in claim.cited_evidence_indexes
    )

    return (
        f"{claim.claim} "
        f"{evidence_markers}"
    )