from math import sqrt
from backend.ingestion.models import PolicyChunk
from backend.retrieval.embeddings import EmbeddingProvider
from backend.retrieval.models import (
    IndexedPolicyChunk,
    RetrievalResult,
)
from backend.retrieval.text import build_retrieval_text

def cosine_similarity(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> float:
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

def build_semantic_index(
    chunks: tuple[PolicyChunk, ...],
    embedding_provider: EmbeddingProvider,
) -> tuple[IndexedPolicyChunk, ...]:
    if not chunks:
        raise ValueError(
            "Cannot build semantic index from an empty chunk collection."
        )

    retrieval_texts = tuple(
        build_retrieval_text(chunk)
        for chunk in chunks
    )

    embeddings = embedding_provider.embed_documents(
        retrieval_texts
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