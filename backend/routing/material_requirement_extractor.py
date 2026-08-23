from typing import Protocol
from backend.routing.material_requirements import (
    MaterialQuestionRequirements,
)

class MaterialRequirementExtractor(
    Protocol
):
    def extract(
        self,
        question: str,
    ) -> MaterialQuestionRequirements:
        ...

def extract_material_requirements(
    question: str,
    extractor: MaterialRequirementExtractor,
) -> MaterialQuestionRequirements:
    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    requirements = extractor.extract(
        question
    )

    if not isinstance(
        requirements,
        MaterialQuestionRequirements,
    ):
        raise ValueError(
            "Material requirement extractor "
            "must return "
            "MaterialQuestionRequirements."
        )

    return requirements