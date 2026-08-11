from backend.core.config import DEFAULT_EVIDENCE_THRESHOLD
from backend.core.models import (
    EvidenceAssessment,
    EvidenceSufficiency,
    RetrievedEvidence,
)

def assess_evidence(
    evidence: tuple[RetrievedEvidence, ...],
    threshold: float | None = None,
) -> EvidenceAssessment:
    if threshold is None:
        threshold = DEFAULT_EVIDENCE_THRESHOLD

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Evidence threshold must be between 0.0 and 1.0.")

    if not evidence:
        return EvidenceAssessment(
            sufficiency=EvidenceSufficiency.INSUFFICIENT,
            best_score=0.0,
            threshold=threshold,
            supporting_evidence=(),
        )

    best_score = max(
        item.relevance_score
        for item in evidence
    )

    supporting_evidence = tuple(
        item
        for item in evidence
        if item.relevance_score >= threshold
    )

    if not supporting_evidence:
        return EvidenceAssessment(
            sufficiency=EvidenceSufficiency.INSUFFICIENT,
            best_score=best_score,
            threshold=threshold,
            supporting_evidence=(),
        )

    return EvidenceAssessment(
        sufficiency=EvidenceSufficiency.SUFFICIENT,
        best_score=best_score,
        threshold=threshold,
        supporting_evidence=supporting_evidence,
    )