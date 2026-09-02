"""
Define the contract for extracting material question requirements.

Material requirements describe significant structural elements that
RAVIN identifies in a parsed question before proposition extraction.
The contract separates requirement extraction from the concrete
dependency-based implementation.

These structures support deterministic question understanding rather
than generated interpretation.
"""

from typing import Protocol
from backend.routing.material_requirements import (
    MaterialRequirementExtractionResult,
)

class MaterialRequirementExtractor(
    Protocol
):
    """Define the contract for extracting material requirements from a question.
    """

    def extract(
        self,
        question: str,
    ) -> MaterialRequirementExtractionResult:
        """Extract the resolved material requirements needed to interpret a question.
        """
        ...

def extract_material_requirements(
    question: str,
    extractor: MaterialRequirementExtractor,
) -> MaterialRequirementExtractionResult:
    """Validate a question and return the configured requirement extractor result.
    """
    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    result = extractor.extract(
        question
    )

    if not isinstance(
        result,
        MaterialRequirementExtractionResult,
    ):
        raise ValueError(
            "Material requirement extractor "
            "must return "
            "MaterialRequirementExtractionResult."
        )

    return result