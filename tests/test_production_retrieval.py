import pytest

from backend.ingestion.models import PolicyChunk
from backend.retrieval.context import (
    ContextAssemblyConfig,
)
from backend.retrieval.production import (
    GroundedRetrievalResult,
    ProductionRetrievalConfig,
    build_production_retrieval_index,
    retrieve_grounded_context,
    retrieve_policy_evidence,
)

class CapturingEmbeddingProvider:
    def __init__(self) -> None:
        self.document_texts: tuple[str, ...] = ()

    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        self.document_texts = texts

        return tuple(
            (1.0, 0.0)
            for _ in texts
        )

    def embed_query(
        self,
        text: str,
    ) -> tuple[float, ...]:
        return (1.0, 0.0)

class CapturingRerankerProvider:
    def __init__(self) -> None:
        self.query: str | None = None
        self.documents: tuple[str, ...] = ()

    def score(
        self,
        query: str,
        documents: tuple[str, ...],
    ) -> tuple[float, ...]:
        self.query = query
        self.documents = documents

        return tuple(
            float(index)
            for index in range(
                1,
                len(documents) + 1,
            )
        )

def make_chunk(
    chunk_index: int,
    text: str | None = None,
) -> PolicyChunk:
    return PolicyChunk(
        policy_id="220",
        policy_title=(
            "Academic Progression Review Policy"
        ),
        source_url=(
            "https://policies.latrobe.edu.au/"
            "document/view.php?id=220"
        ),
        status="Current",
        effective_date=None,
        review_date=None,
        chunk_index=chunk_index,
        text=(
            text
            if text is not None
            else f"Policy evidence {chunk_index}."
        ),
        heading_path=(
            "Section 6 - Procedures",
        ),
    )

def test_production_config_uses_evaluated_defaults():
    config = ProductionRetrievalConfig()

    assert config.top_k == 5
    assert config.rerank_depth == 11
    assert config.semantic_weight == pytest.approx(
        0.85
    )
    assert config.lexical_weight == pytest.approx(
        0.15
    )

def test_production_config_rejects_invalid_top_k():
    with pytest.raises(
        ValueError,
        match=(
            "Production retrieval top_k must be "
            "greater than zero."
        ),
    ):
        ProductionRetrievalConfig(
            top_k=0,
        )

def test_production_config_rejects_invalid_rerank_depth():
    with pytest.raises(
        ValueError,
        match=(
            "Production rerank depth must be "
            "greater than zero."
        ),
    ):
        ProductionRetrievalConfig(
            rerank_depth=0,
        )

def test_production_config_rejects_top_k_above_rerank_depth():
    with pytest.raises(
        ValueError,
        match=(
            "Production retrieval top_k cannot "
            "exceed rerank depth."
        ),
    ):
        ProductionRetrievalConfig(
            top_k=6,
            rerank_depth=5,
        )

def test_production_config_requires_weights_sum_to_one():
    with pytest.raises(
        ValueError,
        match=(
            "Production retrieval weights must "
            "sum to 1."
        ),
    ):
        ProductionRetrievalConfig(
            semantic_weight=0.80,
            lexical_weight=0.30,
        )

def test_production_index_uses_body_only_embeddings():
    provider = CapturingEmbeddingProvider()

    chunk = make_chunk(
        0,
        text="General progression requirements.",
    )

    index = build_production_retrieval_index(
        (chunk,),
        provider,
    )

    assert provider.document_texts == (
        "General progression requirements.",
    )

    assert len(index) == 1
    assert index[0].chunk is chunk

    assert (
        "Academic Progression Review Policy"
        in index[0].retrieval_text
    )

    assert (
        "Section 6 - Procedures"
        in index[0].retrieval_text
    )

def test_production_retrieval_uses_rerank_depth_and_returns_top_k():
    embedding_provider = (
        CapturingEmbeddingProvider()
    )

    reranker_provider = (
        CapturingRerankerProvider()
    )

    chunks = tuple(
        make_chunk(index)
        for index in range(12)
    )

    index = build_production_retrieval_index(
        chunks,
        embedding_provider,
    )

    results = retrieve_policy_evidence(
        index,
        query="academic progression policy",
        embedding_provider=(
            embedding_provider
        ),
        reranker_provider=(
            reranker_provider
        ),
        config=ProductionRetrievalConfig(),
    )

    assert len(
        reranker_provider.documents
    ) == 11

    assert len(results) == 5

    assert [
        result.chunk.chunk_index
        for result in results
    ] == [
        10,
        9,
        8,
        7,
        6,
    ]

    assert results[0].chunk.policy_id == "220"

    assert (
        results[0].chunk.source_url
        == (
            "https://policies.latrobe.edu.au/"
            "document/view.php?id=220"
        )
    )

def test_production_retrieval_rejects_empty_query():
    embedding_provider = (
        CapturingEmbeddingProvider()
    )

    reranker_provider = (
        CapturingRerankerProvider()
    )

    index = build_production_retrieval_index(
        (make_chunk(0),),
        embedding_provider,
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty.",
    ):
        retrieve_policy_evidence(
            index,
            query="   ",
            embedding_provider=(
                embedding_provider
            ),
            reranker_provider=(
                reranker_provider
            ),
            config=(
                ProductionRetrievalConfig()
            ),
        )

def test_grounded_retrieval_returns_structured_result():
    embedding_provider = (
        CapturingEmbeddingProvider()
    )

    reranker_provider = (
        CapturingRerankerProvider()
    )

    chunks = tuple(
        make_chunk(index)
        for index in range(12)
    )

    index = build_production_retrieval_index(
        chunks,
        embedding_provider,
    )

    result = retrieve_grounded_context(
        index,
        query="academic progression policy",
        embedding_provider=(
            embedding_provider
        ),
        reranker_provider=(
            reranker_provider
        ),
        retrieval_config=(
            ProductionRetrievalConfig()
        ),
        context_config=(
            ContextAssemblyConfig()
        ),
    )

    assert isinstance(
        result,
        GroundedRetrievalResult,
    )

    assert len(
        result.retrieval_results
    ) == 5

    assert result.context.evidence_count == 1

    assert result.rendered_context

def test_grounded_retrieval_preserves_ranked_seeds():
    embedding_provider = (
        CapturingEmbeddingProvider()
    )

    reranker_provider = (
        CapturingRerankerProvider()
    )

    chunks = tuple(
        make_chunk(index)
        for index in range(12)
    )

    index = build_production_retrieval_index(
        chunks,
        embedding_provider,
    )

    result = retrieve_grounded_context(
        index,
        query="academic progression policy",
        embedding_provider=(
            embedding_provider
        ),
        reranker_provider=(
            reranker_provider
        ),
        retrieval_config=(
            ProductionRetrievalConfig()
        ),
        context_config=(
            ContextAssemblyConfig()
        ),
    )

    assert [
        item.chunk.chunk_index
        for item in result.retrieval_results
    ] == [
        10,
        9,
        8,
        7,
        6,
    ]

    assert [
        chunk.chunk_index
        for chunk in result.context_chunks[:5]
    ] == [
        10,
        9,
        8,
        7,
        6,
    ]

def test_grounded_retrieval_expands_safe_neighbors():
    embedding_provider = (
        CapturingEmbeddingProvider()
    )

    reranker_provider = (
        CapturingRerankerProvider()
    )

    chunks = tuple(
        make_chunk(index)
        for index in range(12)
    )

    index = build_production_retrieval_index(
        chunks,
        embedding_provider,
    )

    result = retrieve_grounded_context(
        index,
        query="academic progression policy",
        embedding_provider=(
            embedding_provider
        ),
        reranker_provider=(
            reranker_provider
        ),
        retrieval_config=(
            ProductionRetrievalConfig()
        ),
        context_config=(
            ContextAssemblyConfig(
                neighbor_window=1,
                max_context_chunks=15,
            )
        ),
    )

    selected_indexes = {
        chunk.chunk_index
        for chunk in result.context_chunks
    }

    assert {
        6,
        7,
        8,
        9,
        10,
    }.issubset(
        selected_indexes
    )

    assert 5 in selected_indexes
    assert 11 in selected_indexes

def test_grounded_retrieval_renders_citable_evidence():
    embedding_provider = (
        CapturingEmbeddingProvider()
    )

    reranker_provider = (
        CapturingRerankerProvider()
    )

    chunks = tuple(
        make_chunk(index)
        for index in range(12)
    )

    index = build_production_retrieval_index(
        chunks,
        embedding_provider,
    )

    result = retrieve_grounded_context(
        index,
        query="academic progression policy",
        embedding_provider=(
            embedding_provider
        ),
        reranker_provider=(
            reranker_provider
        ),
        retrieval_config=(
            ProductionRetrievalConfig()
        ),
        context_config=(
            ContextAssemblyConfig()
        ),
    )

    assert "[E1]" in (
        result.rendered_context
    )

    assert (
        "Policy ID: 220"
        in result.rendered_context
    )

    assert (
        "Policy Title: "
        "Academic Progression Review Policy"
        in result.rendered_context
    )

    assert (
        "Section 6 - Procedures"
        in result.rendered_context
    )

    assert (
        "https://policies.latrobe.edu.au/"
        "document/view.php?id=220"
        in result.rendered_context
    )