"""
Build and search the in-memory semantic policy index.

This module stores policy chunks with their embedding vectors and
provides cosine-similarity search over the indexed corpus. It contains
the framework-neutral semantic retrieval mechanics used before hybrid
scoring and reranking.

Embedding creation is delegated to an EmbeddingProvider so index logic
does not depend on a specific embedding model.
"""

from math import sqrt
from backend.ingestion.models import PolicyChunk
from backend.retrieval.embeddings import EmbeddingProvider
from backend.retrieval.models import (
    IndexedPolicyChunk,
    RetrievalResult,
)
from backend.retrieval.text import build_retrieval_text

RETRIEVAL_TEXT_EMBEDDING = "retrieval-text"
BODY_ONLY_EMBEDDING = "body-only"
TITLE_BODY_EMBEDDING = "title-body"

def cosine_similarity(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> float:
    """Calculate cosine similarity between two embedding vectors.

    Both vectors must be non-empty, have equal dimensions, and have
    non-zero magnitude. The resulting value measures vector direction
    similarity rather than policy evidence sufficiency.
    """
    if not first or not second:
        raise ValueError(
            "Embedding vectors cannot be empty."
        )

    if len(first) != len(second):
        raise ValueError(
            "Embedding dimensions must match."
        )

    dot_product = sum(
        first_value * second_value
        for first_value, second_value in zip(
            first,
            second,
        )
    )

    first_magnitude = sqrt(
        sum(value * value for value in first)
    )

    second_magnitude = sqrt(
        sum(value * value for value in second)
    )

    if first_magnitude == 0 or second_magnitude == 0:
        raise ValueError(
            "Embedding vectors cannot be zero vectors."
        )

    return dot_product / (
        first_magnitude * second_magnitude
    )

def _build_embedding_text(
    chunk: PolicyChunk,
    strategy: str,
) -> str:
    if strategy == RETRIEVAL_TEXT_EMBEDDING:
        return build_retrieval_text(chunk)

    if strategy == BODY_ONLY_EMBEDDING:
        return chunk.text

    if strategy == TITLE_BODY_EMBEDDING:
        return "\n".join(
            (
                chunk.policy_title,
                chunk.text,
            )
        )

    raise ValueError(
        "Unsupported embedding text strategy."
    )

def build_semantic_index(
    chunks: tuple[PolicyChunk, ...],
    embedding_provider: EmbeddingProvider,
    embedding_text_strategy: str = RETRIEVAL_TEXT_EMBEDDING,
) -> tuple[IndexedPolicyChunk, ...]:
    """Embed policy chunks and construct the in-memory semantic index.

    Retrieval text is retained for later lexical scoring and reranking while
    the configured embedding-text strategy determines what text is sent to
    the embedding provider.

    The provider must return exactly one non-empty vector for each input
    chunk.
    """
    if not chunks:
        raise ValueError(
            "Cannot build semantic index from an empty chunk collection."
        )

    retrieval_texts = tuple(
        build_retrieval_text(chunk)
        for chunk in chunks
    )

    embedding_texts = tuple(
        _build_embedding_text(
            chunk,
            embedding_text_strategy,
        )
        for chunk in chunks
    )

    embeddings = embedding_provider.embed_documents(
        embedding_texts
    )

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Embedding provider returned an unexpected number of vectors."
        )

    indexed_chunks: list[IndexedPolicyChunk] = []

    for chunk, retrieval_text, embedding in zip(
        chunks,
        retrieval_texts,
        embeddings,
    ):
        if not embedding:
            raise ValueError(
                "Embedding vectors cannot be empty."
            )

        indexed_chunks.append(
            IndexedPolicyChunk(
                chunk=chunk,
                retrieval_text=retrieval_text,
                embedding=embedding,
            )
        )

    return tuple(indexed_chunks)

def search_semantic_index(
    indexed_chunks: tuple[IndexedPolicyChunk, ...],
    query: str,
    embedding_provider: EmbeddingProvider,
    top_k: int = 5,
) -> tuple[RetrievalResult, ...]:
    """Rank indexed policy chunks by query-to-chunk cosine similarity.

    The query is embedded once and compared with each stored chunk vector.
    Results are returned in descending similarity order up to top_k.

    This function provides semantic-only retrieval and does not perform the
    production hybrid or reranking stages.
    """
    query = query.strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if not indexed_chunks:
        raise ValueError(
            "Cannot search an empty semantic index."
        )

    query_embedding = embedding_provider.embed_query(
        query
    )

    results = [
        RetrievalResult(
            chunk=indexed_chunk.chunk,
            score=cosine_similarity(
                query_embedding,
                indexed_chunk.embedding,
            ),
        )
        for indexed_chunk in indexed_chunks
    ]

    results.sort(
        key=lambda result: result.score,
        reverse=True,
    )

    return tuple(results[:top_k])