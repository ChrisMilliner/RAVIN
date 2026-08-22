from dataclasses import dataclass
from backend.ingestion.models import PolicyChunk
from backend.retrieval.embeddings import (
    EmbeddingProvider,
)
from backend.retrieval.hybrid import (
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_SEMANTIC_WEIGHT,
    search_hybrid_index,
)
from backend.retrieval.index import (
    BODY_ONLY_EMBEDDING,
    build_semantic_index,
)
from backend.retrieval.models import (
    IndexedPolicyChunk,
    RetrievalResult,
)
from backend.retrieval.reranking import (
    RerankerProvider,
    rerank_results,
)

DEFAULT_PRODUCTION_TOP_K = 5
DEFAULT_PRODUCTION_RERANK_DEPTH = 11

@dataclass(frozen=True)
class ProductionRetrievalConfig:
    top_k: int = DEFAULT_PRODUCTION_TOP_K
    rerank_depth: int = (
        DEFAULT_PRODUCTION_RERANK_DEPTH
    )
    semantic_weight: float = (
        DEFAULT_SEMANTIC_WEIGHT
    )
    lexical_weight: float = (
        DEFAULT_LEXICAL_WEIGHT
    )

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError(
                "Production retrieval top_k must be "
                "greater than zero."
            )

        if self.rerank_depth <= 0:
            raise ValueError(
                "Production rerank depth must be "
                "greater than zero."
            )

        if self.top_k > self.rerank_depth:
            raise ValueError(
                "Production retrieval top_k cannot "
                "exceed rerank depth."
            )

        if self.semantic_weight < 0.0:
            raise ValueError(
                "Production semantic weight cannot "
                "be negative."
            )

        if self.lexical_weight < 0.0:
            raise ValueError(
                "Production lexical weight cannot "
                "be negative."
            )

        if abs(
            self.semantic_weight
            + self.lexical_weight
            - 1.0
        ) > 1e-9:
            raise ValueError(
                "Production retrieval weights must "
                "sum to 1."
            )

def build_production_retrieval_index(
    chunks: tuple[PolicyChunk, ...],
    embedding_provider: EmbeddingProvider,
) -> tuple[IndexedPolicyChunk, ...]:
    return build_semantic_index(
        chunks,
        embedding_provider,
        embedding_text_strategy=(
            BODY_ONLY_EMBEDDING
        ),
    )

def retrieve_policy_evidence(
    indexed_chunks: tuple[
        IndexedPolicyChunk,
        ...
    ],
    query: str,
    embedding_provider: EmbeddingProvider,
    reranker_provider: RerankerProvider,
    config: ProductionRetrievalConfig,
) -> tuple[RetrievalResult, ...]:
    hybrid_results = search_hybrid_index(
        indexed_chunks,
        query=query,
        embedding_provider=embedding_provider,
        top_k=config.rerank_depth,
        semantic_weight=config.semantic_weight,
        lexical_weight=config.lexical_weight,
    )

    reranked_results = rerank_results(
        query=query,
        results=hybrid_results,
        reranker_provider=reranker_provider,
    )

    return reranked_results[:config.top_k]