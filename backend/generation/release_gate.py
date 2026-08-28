from dataclasses import dataclass
from backend.generation.citation_validator import (
    validate_generation_citations,
)
from backend.generation.grounded_generator import (
    GroundedAnswerGenerator,
    GroundedGenerationRequest,
    generate_grounded_answer,
)

class GroundedGenerationRejectedError(
    RuntimeError
):
    pass

@dataclass(frozen=True)
class ReleasedGroundedAnswer:
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
) -> ReleasedGroundedAnswer:
    generation_result = (
        generate_grounded_answer(
            request,
            generator,
        )
    )

    validation = (
        validate_generation_citations(
            request,
            generation_result,
        )
    )

    if not validation.valid:
        raise GroundedGenerationRejectedError(
            validation.reason
        )

    return ReleasedGroundedAnswer(
        text=generation_result.text,
        cited_evidence_indexes=(
            validation.cited_evidence_indexes
        ),
    )