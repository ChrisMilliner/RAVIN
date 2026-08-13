from dataclasses import dataclass
from backend.ingestion.models import PolicyChunk

@dataclass(frozen=True)
class IndexedPolicyChunk:
    chunk: PolicyChunk
    retrieval_text: str
    embedding: tuple[float, ...]

@dataclass(frozen=True)
class RetrievalResult:
    chunk: PolicyChunk
    score: float