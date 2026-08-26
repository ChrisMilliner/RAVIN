from typing import Protocol
from backend.routing.material_propositions import (
    MaterialQuestionPropositions,
)
from backend.routing.material_requirements import (
    MaterialQuestionRequirements,
)
from backend.routing.question_parser import (
    QuestionParse,
)

class MaterialPropositionExtractor(
    Protocol
):
    def extract(
        self,
        question: str,
        requirements: MaterialQuestionRequirements,
        parse: QuestionParse,
    ) -> MaterialQuestionPropositions:
        ...

def extract_material_propositions(
    question: str,
    requirements: MaterialQuestionRequirements,
    parse: QuestionParse,
    extractor: MaterialPropositionExtractor,
) -> MaterialQuestionPropositions:
    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    if not isinstance(
        requirements,
        MaterialQuestionRequirements,
    ):
        raise ValueError(
            "Requirements must be "
            "MaterialQuestionRequirements."
        )

    if not isinstance(
        parse,
        QuestionParse,
    ):
        raise ValueError(
            "Parse must be a QuestionParse."
        )

    result = extractor.extract(
        question,
        requirements,
        parse,
    )

    if not isinstance(
        result,
        MaterialQuestionPropositions,
    ):
        raise ValueError(
            "Material proposition extractor "
            "must return "
            "MaterialQuestionPropositions."
        )

    return result