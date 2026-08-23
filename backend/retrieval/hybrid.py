import re
from backend.retrieval.embeddings import EmbeddingProvider
from backend.retrieval.index import cosine_similarity
from backend.retrieval.models import (
    IndexedPolicyChunk,
    RetrievalResult,
)

DEFAULT_SEMANTIC_WEIGHT = 0.85
DEFAULT_LEXICAL_WEIGHT = 0.15

STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "does",
        "for",
        "from",
        "how",
        "if",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "their",
        "they",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)

def tokenize_lexical_text(
    text: str,
) -> frozenset[str]:
    tokens = re.findall(
        r"[a-z0-9]+",
        text.casefold(),
    )

    return frozenset(
        token
        for token in tokens
        if token not in STOP_WORDS
    )

def calculate_lexical_coverage(
    query: str,
    retrieval_text: str,
) -> float:
    query_tokens = tokenize_lexical_text(
        query
    )

    if not query_tokens:
        return 0.0

    document_tokens = tokenize_lexical_text(
        retrieval_text
    )

    matching_tokens = (
        query_tokens & document_tokens
    )

    return (
        len(matching_tokens)
        / len(query_tokens)
    )

def calculate_hybrid_score(
    semantic_score: float,
    lexical_score: float,
    semantic_weight: float = (
        DEFAULT_SEMANTIC_WEIGHT
    ),
    lexical_weight: float = (
        DEFAULT_LEXICAL_WEIGHT
    ),
) -> float:
    if semantic_weight < 0.0:
        raise ValueError(
            "Semantic weight cannot be negative."
        )

    if lexical_weight < 0.0:
        raise ValueError(
            "Lexical weight cannot be negative."
        )

    total_weight = (
        semantic_weight + lexical_weight
    )

    if total_weight == 0.0:
        raise ValueError(
            "Hybrid retrieval weights cannot both be zero."
        )

    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError(
            "Hybrid retrieval weights must sum to 1."
        )

    return (
        semantic_score * semantic_weight
        + lexical_score * lexical_weight
    )

def search_hybrid_index(
    indexed_chunks: tuple[IndexedPolicyChunk, ...],
    query: str,
    embedding_provider: EmbeddingProvider,
    top_k: int = 5,
    semantic_weight: float = (
        DEFAULT_SEMANTIC_WEIGHT
    ),
    lexical_weight: float = (
        DEFAULT_LEXICAL_WEIGHT
    ),
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

    query_embedding = (
        embedding_provider.embed_query(
            query
        )
    )

    results: list[RetrievalResult] = []

    for indexed_chunk in indexed_chunks:
        semantic_score = cosine_similarity(
            query_embedding,
            indexed_chunk.embedding,
        )

        lexical_score = (
            calculate_lexical_coverage(
                query,
                indexed_chunk.retrieval_text,
            )
        )

        hybrid_score = calculate_hybrid_score(
            semantic_score,
            lexical_score,
            semantic_weight,
            lexical_weight,
        )

        results.append(
            RetrievalResult(
                chunk=indexed_chunk.chunk,
                score=hybrid_score,
            )
        )

    results.sort(
        key=lambda result: result.score,
        reverse=True,
    )

    return tuple(results[:top_k])