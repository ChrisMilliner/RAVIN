import pytest

from backend.ingestion.models import PolicyChunk
from backend.retrieval.production import (
    ProductionRetrievalConfig,
    build_production_retrieval_index,
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