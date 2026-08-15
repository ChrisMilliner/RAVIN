from typing import Protocol
from backend.retrieval.models import (
    RetrievalResult,
)
from backend.retrieval.text import (
    build_retrieval_text,
)

class RerankerProvider(Protocol):
    def score(
        self,
        query: str,
        documents: tuple[str, ...],
    ) -> tuple[float, ...]:
        ...

def rerank_results(
    query: str,
    results: tuple[RetrievalResult, ...],
    reranker_provider: RerankerProvider,
) -> tuple[RetrievalResult, ...]:
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