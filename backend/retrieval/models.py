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
    """Associate a policy chunk with its retrieval text and embedding vector.

    The model keeps the original traceable PolicyChunk alongside the
    representations required by semantic, lexical, and reranking stages.
    """

    chunk: PolicyChunk
    retrieval_text: str
    embedding: tuple[float, ...]

@dataclass(frozen=True)
class RetrievalResult:
    """Represent one ranked policy chunk and its current retrieval score.

    The meaning of score depends on the retrieval stage that produced the
    result, such as semantic, hybrid, or reranker scoring.
    """

    chunk: PolicyChunk
    score: float