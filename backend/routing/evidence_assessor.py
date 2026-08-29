"""
Define the framework-neutral evidence-sufficiency assessment contract.

Evidence sufficiency represents whether retrieved policy evidence is
adequate to support all material parts of a clear user question. The
contract separates this control decision from the concrete assessment
strategy.

RAVIN distinguishes insufficient evidence from ambiguity: a clear
question with inadequate evidence is not the same as a question that
requires clarification.
"""

from typing import Protocol
from backend.retrieval.production import (
    GroundedRetrievalResult,
)
from backend.routing.models import (
    EvidenceAssessment,
    QuestionIntent,
)

class EvidenceSufficiencyAssessor(
    Protocol
):
    def assess(
        self,
        question: str,
        intent: QuestionIntent,
        retrieval_result: (
            GroundedRetrievalResult
        ),
    ) -> EvidenceAssessment:
        ...

def assess_evidence_sufficiency(
    question: str,
    intent: QuestionIntent,
    retrieval_result: (
        GroundedRetrievalResult
    ),
    assessor: EvidenceSufficiencyAssessor,
) -> EvidenceAssessment:
    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    if not isinstance(
        intent,
        QuestionIntent,
    ):
        raise ValueError(
            "Question intent must be a "
            "QuestionIntent."
        )

    if intent == QuestionIntent.AMBIGUOUS:
        raise ValueError(
            "Evidence sufficiency must not be "
            "assessed for an ambiguous question."
        )

    if not isinstance(
        retrieval_result,
        GroundedRetrievalResult,
    ):
        raise ValueError(
            "Retrieval result must be a "
            "GroundedRetrievalResult."
        )

    assessment = assessor.assess(
        question,
        intent,
        retrieval_result,
    )

    if not isinstance(
        assessment,
        EvidenceAssessment,
    ):
        raise ValueError(
            "Evidence assessor must return an "
            "EvidenceAssessment."
        )

    return assessment