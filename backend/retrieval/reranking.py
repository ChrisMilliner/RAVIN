"""
Define reranking contracts and candidate reranking behaviour.

This module provides the neutral reranker interface and the logic for
applying reranker scores to an existing retrieval candidate set.
Reranking improves the ordering of evidence after initial retrieval.

Concrete cross-encoder technology is supplied through a provider
adapter rather than embedded in the ranking algorithm.
"""

from typing import Protocol
from backend.retrieval.models import (
    RetrievalResult,
)
from backend.retrieval.text import (
    build_retrieval_text,
)

class RerankerProvider(Protocol):
    """Define the framework-neutral candidate reranking contract.

    Implementations score an ordered collection of candidate documents
    against one query without exposing model-specific APIs to retrieval
    business logic.
    """

    def score(
        self,
        query: str,
        documents: tuple[str, ...],
    ) -> tuple[float, ...]:
        """Return one relevance score for each candidate document.

        Scores must correspond positionally to the supplied document collection.
        """
        ...

def rerank_results(
    query: str,
    results: tuple[RetrievalResult, ...],
    reranker_provider: RerankerProvider,
) -> tuple[RetrievalResult, ...]:
    """Reorder retrieval candidates using the configured reranker provider.

    Candidate PolicyChunks are converted to full retrieval text, scored
    against the query, and rebuilt as RetrievalResults ordered by descending
    reranker score.

    A mismatched provider score count is rejected rather than silently
    producing an incomplete ranking.
    """
    query = query.strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    if not results:
        raise ValueError(
            "Cannot rerank an empty result set."
        )

    documents = tuple(
        build_retrieval_text(
            result.chunk
        )
        for result in results
    )

    scores = reranker_provider.score(
        query,
        documents,
    )

    if len(scores) != len(results):
        raise ValueError(
            "Reranker returned an unexpected "
            "number of scores."
        )

    reranked_results = [
        RetrievalResult(
            chunk=result.chunk,
            score=score,
        )
        for result, score in zip(
            results,
            scores,
        )
    ]

    reranked_results.sort(
        key=lambda result: result.score,
        reverse=True,
    )

    return tuple(reranked_results)