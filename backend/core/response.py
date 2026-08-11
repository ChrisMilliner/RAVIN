from backend.core.evidence import assess_evidence
from backend.core.messages import INSUFFICIENT_EVIDENCE_MESSAGE
from backend.core.models import (
    EvidenceSufficiency,
    GroundedResponse,
    PolicyDocument,
    ResponseOutcome,
)
from backend.core.retrieval import retrieve_evidence

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "RAVIN could not find sufficient policy evidence to provide "
    "a supported answer."
)

def build_grounded_response(
    question: str,
    policies: tuple[PolicyDocument, ...],
    threshold: float | None = None,
) -> GroundedResponse:
    if not question.strip():
        raise ValueError("Question must contain non-whitespace text.")

    evidence = retrieve_evidence(
        question,
        policies,
    )

    assessment = assess_evidence(
        evidence,
        threshold=threshold,
    )

    if assessment.sufficiency is EvidenceSufficiency.INSUFFICIENT:
        return GroundedResponse(
            outcome=ResponseOutcome.INSUFFICIENT_EVIDENCE,
            answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            sources=(),
        )

    supporting_evidence = assessment.supporting_evidence

    primary_evidence = supporting_evidence[0]

    answer = (
        "Reference answer based on retrieved evidence: "
        f"{primary_evidence.text}"
    )

    return GroundedResponse(
        outcome=ResponseOutcome.SUPPORTED,
        answer=answer,
        sources=supporting_evidence,
    )