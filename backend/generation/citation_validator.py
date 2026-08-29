"""
Validate evidence citations attached to generated RAVIN answers.

This module performs deterministic checks on the internal evidence
markers produced during grounded generation. It verifies that factual
output cites supplied evidence and that cited indexes refer to evidence
actually available to the generator.

Citation validation forms part of the fail-closed release path and does
not rely on a language model to decide whether citations are valid.
"""

import re
from dataclasses import dataclass
from backend.generation.grounded_generator import (
    GroundedGenerationRequest,
    GroundedGenerationResult,
)

_EVIDENCE_MARKER_PATTERN = re.compile(
    r"\[E(\d+)\]"
)

@dataclass(frozen=True)
class CitationValidationResult:
    valid: bool
    cited_evidence_indexes: tuple[int, ...]
    reason: str

    def __post_init__(self) -> None:
        if any(
            index < 1
            for index in self.cited_evidence_indexes
        ):
            raise ValueError(
                "Cited evidence indexes must "
                "be positive."
            )

        if not self.reason.strip():
            raise ValueError(
                "Citation validation reason "
                "cannot be empty."
            )

def validate_generation_citations(
    request: GroundedGenerationRequest,
    result: GroundedGenerationResult,
) -> CitationValidationResult:
    if not isinstance(
        request,
        GroundedGenerationRequest,
    ):
        raise ValueError(
            "Citation validation request must be "
            "a GroundedGenerationRequest."
        )

    if not isinstance(
        result,
        GroundedGenerationResult,
    ):
        raise ValueError(
            "Citation validation result must be "
            "a GroundedGenerationResult."
        )

    cited_indexes = _extract_cited_indexes(
        result.text
    )

    if not cited_indexes:
        return CitationValidationResult(
            valid=False,
            cited_evidence_indexes=(),
            reason=(
                "Generated answer does not cite "
                "any approved evidence."
            ),
        )

    maximum_index = len(
        request.evidence_texts
    )

    invalid_indexes = tuple(
        index
        for index in cited_indexes
        if index > maximum_index
    )

    if invalid_indexes:
        return CitationValidationResult(
            valid=False,
            cited_evidence_indexes=(
                cited_indexes
            ),
            reason=(
                "Generated answer cites evidence "
                "that was not supplied."
            ),
        )

    return CitationValidationResult(
        valid=True,
        cited_evidence_indexes=(
            cited_indexes
        ),
        reason=(
            "All generated evidence citations "
            "refer to supplied evidence."
        ),
    )

def _extract_cited_indexes(
    text: str,
) -> tuple[int, ...]:
    matches = (
        _EVIDENCE_MARKER_PATTERN.findall(
            text
        )
    )

    indexes = tuple(
        int(match)
        for match in matches
    )

    unique_indexes = tuple(
        dict.fromkeys(
            indexes
        )
    )

    return unique_indexes