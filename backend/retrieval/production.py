"""
Compose the production policy retrieval and context-building pipeline.

This module applies the configured semantic search, hybrid scoring,
reranking depth, result limits, and grounded-context assembly used by
RAVIN's application service.

The production retrieval result contains both ranked policy evidence
and traceable context for downstream evidence assessment. Retrieval
ranking alone is not treated as proof that a question can be answered.
"""

from dataclasses import dataclass
from backend.ingestion.models import PolicyChunk
from backend.retrieval.context import (
    ContextAssemblyConfig,
    GroundedContext,
    assemble_context_chunks,
    build_grounded_context,
    render_grounded_context,
)
from backend.retrieval.embeddings import (
    EmbeddingProvider,
)
from backend.retrieval.hybrid import (
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_SEMANTIC_WEIGHT,
    search_hybrid_index,
    tokenize_lexical_text,
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
DEFAULT_PRODUCTION_RERANK_DEPTH = 50
_GENERIC_POLICY_TITLE_TOKENS = frozenset(
    {
        "policy",
        "procedure",
        "procedures",
        "standard",
        "standards",
        "guideline",
        "guidelines",
        "charter",
        "schedule",
        "student",
        "students",
        "staff",
        "university",
        "la",
        "trobe",
    }
)

_MIN_RELATED_TOKEN_LENGTH = 6
_MIN_TITLE_ANCHOR_MATCH_COUNT = 2
_MIN_SINGLE_TITLE_COVERAGE = 0.75

def _title_tokens_are_related(
    left: str,
    right: str,
) -> bool:
    if left == right:
        return True

    shorter_length = min(
        len(left),
        len(right),
    )

    if (
        shorter_length
        < _MIN_RELATED_TOKEN_LENGTH
    ):
        return False

    return (
        left.startswith(right)
        or right.startswith(left)
    )

def _policy_title_match_signal(
    query: str,
    policy_title: str,
) -> tuple[
    bool,
    int,
    float,
]:
    query_tokens = tuple(
        tokenize_lexical_text(
            query
        )
    )

    title_tokens = tuple(
        token
        for token in tokenize_lexical_text(
            policy_title
        )
        if token
        not in _GENERIC_POLICY_TITLE_TOKENS
    )

    if not title_tokens:
        return (
            False,
            0,
            0.0,
        )

    matched_title_tokens = {
        title_token
        for title_token in title_tokens
        if any(
            _title_tokens_are_related(
                query_token,
                title_token,
            )
            for query_token
            in query_tokens
        )
    }

    match_count = len(
        matched_title_tokens
    )

    coverage = (
        match_count
        / len(title_tokens)
    )

    anchored = (
        match_count
        >= _MIN_TITLE_ANCHOR_MATCH_COUNT
        or (
            match_count == 1
            and coverage
            >= _MIN_SINGLE_TITLE_COVERAGE
        )
    )

    return (
        anchored,
        match_count,
        coverage,
    )

def prioritize_policy_title_matches(
    query: str,
    results: tuple[
        RetrievalResult,
        ...
    ],
) -> tuple[
    RetrievalResult,
    ...
]:
    """Prioritize strong policy-title matches without discarding reranker order.

    Strong title matches are promoted ahead of generic cross-encoder matches.
    Weak or partial title matches do not affect the existing reranker order.
    """
    query = query.strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    if not results:
        raise ValueError(
            "Cannot prioritize an empty result set."
        )

    def ranking_key(
        result: RetrievalResult,
    ) -> tuple[
        int,
        int,
        float,
        float,
    ]:
        (
            anchored,
            match_count,
            coverage,
        ) = _policy_title_match_signal(
            query,
            result.chunk.policy_title,
        )

        if not anchored:
            return (
                0,
                0,
                0.0,
                result.score,
            )

        return (
            1,
            match_count,
            coverage,
            result.score,
        )

    prioritized = sorted(
        results,
        key=ranking_key,
        reverse=True,
    )

    return tuple(
        prioritized
    )

@dataclass(frozen=True)
class ProductionRetrievalConfig:
    """Configure the candidate, reranking, and final limits used in production.

    rerank_depth controls how many hybrid candidates reach the reranker,
    while top_k controls how many reranked results are retained. Semantic
    and lexical weights configure hybrid candidate scoring.
    """

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

@dataclass(frozen=True)
class GroundedRetrievalResult:
    """Represent the complete result of production grounded retrieval.

    The result retains final ranked evidence, the structurally expanded
    context chunks, structured GroundedContext, and rendered evidence text
    used by later RAVIN stages.
    """

    retrieval_results: tuple[
        RetrievalResult,
        ...
    ]
    context_chunks: tuple[
        PolicyChunk,
        ...
    ]
    context: GroundedContext
    rendered_context: str

def build_production_retrieval_index(
    chunks: tuple[PolicyChunk, ...],
    embedding_provider: EmbeddingProvider,
) -> tuple[IndexedPolicyChunk, ...]:
    """Build the semantic index using the production embedding strategy.

    Production embeddings use policy body text only. The index still
    retains full retrieval text containing title, heading path, and body for
    later lexical scoring and reranking.
    """
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
    """Run the production candidate retrieval and reranking sequence.

    Hybrid retrieval first selects rerank_depth candidates using semantic
    and lexical signals. The configured reranker then reorders those
    candidates and only the final top_k results are returned.

    The resulting rank is evidence discovery, not a sufficiency decision.
    """
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

    prioritized_results = (
        prioritize_policy_title_matches(
            query,
            reranked_results,
        )
    )

    return prioritized_results[
        :config.top_k
    ]

def retrieve_grounded_context(
    indexed_chunks: tuple[
        IndexedPolicyChunk,
        ...
    ],
    query: str,
    embedding_provider: EmbeddingProvider,
    reranker_provider: RerankerProvider,
    retrieval_config: ProductionRetrievalConfig,
    context_config: ContextAssemblyConfig,
) -> GroundedRetrievalResult:
    """Retrieve ranked policy evidence and assemble its grounded context.

    The function performs production evidence retrieval, expands selected
    chunks with bounded structural neighbours, builds traceable context
    blocks, and renders those blocks for downstream assessment and
    generation.

    The result contains both ranked seeds and the expanded context so later
    stages can retain retrieval provenance.
    """
    retrieval_results = retrieve_policy_evidence(
        indexed_chunks,
        query=query,
        embedding_provider=embedding_provider,
        reranker_provider=reranker_provider,
        config=retrieval_config,
    )

    corpus_chunks = tuple(
        indexed_chunk.chunk
        for indexed_chunk in indexed_chunks
    )

    context_chunks = assemble_context_chunks(
        corpus_chunks,
        retrieval_results,
        context_config,
    )

    context = build_grounded_context(
        context_chunks,
    )

    rendered_context = render_grounded_context(
        context,
    )

    return GroundedRetrievalResult(
        retrieval_results=retrieval_results,
        context_chunks=context_chunks,
        context=context,
        rendered_context=rendered_context,
    )