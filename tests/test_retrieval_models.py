from backend.ingestion.models import PolicyChunk
from backend.retrieval.models import (
    IndexedPolicyChunk,
    RetrievalResult,
)

def make_chunk() -> PolicyChunk:
    return PolicyChunk(
        policy_id="208",
        policy_title="Academic Dress Policy",
        source_url=(
            "https://policies.latrobe.edu.au/"
            "document/view.php?id=208"
        ),
        status="Current",
        effective_date="14th November 2025",
        review_date="13th November 2028",
        chunk_index=7,
        text="Requests for changes require approval.",
        heading_path=(
            "Section 6 - Procedures",
            "Part B - Requests for Changes",
        ),
    )

def test_indexed_policy_chunk_preserves_original_chunk():
    chunk = make_chunk()

    indexed = IndexedPolicyChunk(
        chunk=chunk,
        retrieval_text=(
            "Academic Dress Policy "
            "Section 6 - Procedures "
            "Part B - Requests for Changes "
            "Requests for changes require approval."
        ),
        embedding=(0.1, 0.2, 0.3),
    )

    assert indexed.chunk is chunk
    assert indexed.embedding == (0.1, 0.2, 0.3)

def test_retrieval_result_preserves_chunk_and_score():
    chunk = make_chunk()

    result = RetrievalResult(
        chunk=chunk,
        score=0.82,
    )

    assert result.chunk is chunk
    assert result.score == 0.82