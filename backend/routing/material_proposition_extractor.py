"""
Define the contract for extracting material propositions from questions.

A material proposition represents a substantive factual requirement
that retrieved evidence must support before RAVIN can safely answer the
question.

The extraction contract keeps evidence-requirement modelling separate
from parser implementations and proposition-coverage scoring.
"""

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
    """Define the contract for extracting material propositions from a question.
    """

    def extract(
        self,
        question: str,
        requirements: MaterialQuestionRequirements,
        parse: QuestionParse,
    ) -> MaterialQuestionPropositions:
        """Extract propositions from a question, its requirements, and resolved parse.
        """
        ...

def extract_material_propositions(
    question: str,
    requirements: MaterialQuestionRequirements,
    parse: QuestionParse,
    extractor: MaterialPropositionExtractor,
) -> MaterialQuestionPropositions:
    """Validate proposition-extraction inputs and return the extractor result.
    """
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