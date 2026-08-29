"""
Define the data models shared by RAVIN retrieval components.

These models associate policy chunks with vector-index information,
retrieval scores, and ranked retrieval results. They carry the source
data required by later context assembly and evidence tracing.

The models represent retrieval state and do not make routing or answer
generation decisions.
"""

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