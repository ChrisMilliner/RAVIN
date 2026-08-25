from typing import Protocol
from backend.routing.material_requirements import (
    MaterialRequirementExtractionResult,
)

class MaterialRequirementExtractor(
    Protocol
):
    def extract(
        self,
        question: str,
    ) -> MaterialRequirementExtractionResult:
        ...

def extract_material_requirements(
    question: str,
    extractor: MaterialRequirementExtractor,
) -> MaterialRequirementExtractionResult:
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