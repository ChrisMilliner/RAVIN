from dataclasses import dataclass
from enum import Enum

class ResponseOutcome(str, Enum):
    SUPPORTED = "supported"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"

class EvidenceSufficiency(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"

@dataclass(frozen=True)
class PolicyDocument:
    policy_id: str
    title: str
    source_url: str
    status: str
    text: str

@dataclass(frozen=True)
class RetrievedEvidence:
    policy_id: str
    policy_title: str
    source_url: str
    text: str
    relevance_score: float

@dataclass(frozen=True)
class EvidenceAssessment:
    sufficiency: EvidenceSufficiency
    best_score: float
    threshold: float
    supporting_evidence: tuple[RetrievedEvidence, ...]

@dataclass(frozen=True)
class GroundedResponse:
    outcome: ResponseOutcome
    answer: str | None
    sources: tuple[RetrievedEvidence, ...]