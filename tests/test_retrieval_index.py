from backend.ingestion.models import PolicyChunk
from backend.retrieval.index import (
    TITLE_BODY_EMBEDDING,
    build_semantic_index,
    cosine_similarity,
    search_semantic_index,
)
from backend.retrieval.text import build_retrieval_text

class FakeEmbeddingProvider:
    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            self._embed(text)
            for text in texts
        )

    def embed_query(
        self,
        text: str,
    ) -> tuple[float, ...]:
        return self._embed(text)

    def _embed(
        self,
        text: str,
    ) -> tuple[float, ...]:
        lowered = text.casefold()

        return (
            1.0 if "dress" in lowered else 0.0,
            1.0 if "admission" in lowered else 0.0,
            1.0 if "extension" in lowered else 0.0,
        )

def make_chunk(
    policy_id: str,
    title: str,
    text: str,
    heading_path: tuple[str, ...],
) -> PolicyChunk:
    return PolicyChunk(
        policy_id=policy_id,
        policy_title=title,
        source_url=(
            "https://policies.latrobe.edu.au/"
            f"document/view.php?id={policy_id}"
        ),
        status="Current",
        effective_date=None,
        review_date=None,
        chunk_index=0,
        text=text,
        heading_path=heading_path,
    )

def test_build_retrieval_text_includes_policy_context():
    chunk = make_chunk(
        policy_id="208",
        title="Academic Dress Policy",
        text="Requests require approval.",
        heading_path=(
            "Section 6 - Procedures",
            "Part B - Requests",
        ),
    )

    retrieval_text = build_retrieval_text(chunk)

    assert "Academic Dress Policy" in retrieval_text
    assert "Section 6 - Procedures" in retrieval_text
    assert "Part B - Requests" in retrieval_text
    assert "Requests require approval." in retrieval_text

def test_cosine_similarity_identical_vectors_is_one():
    score = cosine_similarity(
        (1.0, 2.0, 3.0),
        (1.0, 2.0, 3.0),
    )

    assert score == 1.0

def test_build_semantic_index_preserves_chunk_and_context():
    provider = FakeEmbeddingProvider()

    chunk = make_chunk(
        policy_id="208",
        title="Academic Dress Policy",
        text="Academic dress requirements.",
        heading_path=("Section 6 - Procedures",),
    )

    index = build_semantic_index(
        (chunk,),
        provider,
    )

    assert len(index) == 1
    assert index[0].chunk is chunk
    assert "Academic Dress Policy" in index[0].retrieval_text
    assert index[0].embedding == (1.0, 0.0, 0.0)

def test_build_semantic_index_can_embed_body_only():
    provider = FakeEmbeddingProvider()

    chunk = make_chunk(
        policy_id="208",
        title="Academic Dress Policy",
        text="General requirements.",
        heading_path=(
            "Section 6 - Procedures",
        ),
    )

    index = build_semantic_index(
        (chunk,),
        provider,
        embedding_text_strategy="body-only",
    )

    assert len(index) == 1

    assert (
        "Academic Dress Policy"
        in index[0].retrieval_text
    )

    assert (
        "Section 6 - Procedures"
        in index[0].retrieval_text
    )

    assert (
        index[0].embedding
        == (0.0, 0.0, 0.0)
    )

def test_build_semantic_index_can_embed_title_and_body_without_heading():
    provider = FakeEmbeddingProvider()

    chunk = make_chunk(
        policy_id="208",
        title="Academic Dress Policy",
        text="Extension requirements.",
        heading_path=(
            "Admission Section",
        ),
    )

    index = build_semantic_index(
        (chunk,),
        provider,
        embedding_text_strategy=(
            TITLE_BODY_EMBEDDING
        ),
    )

    assert len(index) == 1

    assert (
        "Academic Dress Policy"
        in index[0].retrieval_text
    )

    assert (
        "Admission Section"
        in index[0].retrieval_text
    )

    assert (
        index[0].embedding
        == (1.0, 0.0, 1.0)
    )

def test_build_semantic_index_rejects_unknown_embedding_strategy():
    provider = FakeEmbeddingProvider()

    chunk = make_chunk(
        policy_id="208",
        title="Academic Dress Policy",
        text="Academic dress requirements.",
        heading_path=(
            "Section 6 - Procedures",
        ),
    )

    try:
        build_semantic_index(
            (chunk,),
            provider,
            embedding_text_strategy="unknown",
        )
    except ValueError as exc:
        assert str(exc) == (
            "Unsupported embedding text strategy."
        )
    else:
        raise AssertionError(
            "Expected unknown embedding "
            "strategy to raise ValueError."
        )

def test_semantic_search_ranks_relevant_chunk_first():
    provider = FakeEmbeddingProvider()

    dress_chunk = make_chunk(
        policy_id="208",
        title="Academic Dress Policy",
        text="Academic dress requirements.",
        heading_path=("Section 6 - Procedures",),
    )

    admissions_chunk = make_chunk(
        policy_id="169",
        title="Admissions Policy",
        text="Admission requirements.",
        heading_path=("Section 5 - Policy Statement",),
    )

    index = build_semantic_index(
        (
            admissions_chunk,
            dress_chunk,
        ),
        provider,
    )

    results = search_semantic_index(
        index,
        query="What are the academic dress requirements?",
        embedding_provider=provider,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk is dress_chunk
    assert results[0].score > results[1].score

def test_semantic_search_respects_top_k():
    provider = FakeEmbeddingProvider()

    chunks = (
        make_chunk(
            "208",
            "Academic Dress Policy",
            "Dress requirements.",
            ("Section 1",),
        ),
        make_chunk(
            "169",
            "Admissions Policy",
            "Admission requirements.",
            ("Section 1",),
        ),
    )

    index = build_semantic_index(
        chunks,
        provider,
    )

    results = search_semantic_index(
        index,
        query="dress",
        embedding_provider=provider,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.policy_id == "208"

def test_semantic_search_rejects_empty_query():
    provider = FakeEmbeddingProvider()

    chunk = make_chunk(
        "208",
        "Academic Dress Policy",
        "Dress requirements.",
        ("Section 1",),
    )

    index = build_semantic_index(
        (chunk,),
        provider,
    )

    try:
        search_semantic_index(
            index,
            query="   ",
            embedding_provider=provider,
        )
    except ValueError as exc:
        assert str(exc) == "Query cannot be empty."
    else:
        raise AssertionError(
            "Expected empty query to raise ValueError."
        )

def test_semantic_search_rejects_invalid_top_k():
    provider = FakeEmbeddingProvider()

    chunk = make_chunk(
        "208",
        "Academic Dress Policy",
        "Dress requirements.",
        ("Section 1",),
    )

    index = build_semantic_index(
        (chunk,),
        provider,
    )

    try:
        search_semantic_index(
            index,
            query="dress",
            embedding_provider=provider,
            top_k=0,
        )
    except ValueError as exc:
        assert str(exc) == "top_k must be greater than zero."
    else:
        raise AssertionError(
            "Expected invalid top_k to raise ValueError."
        )

def test_build_semantic_index_rejects_empty_corpus():
    provider = FakeEmbeddingProvider()

    try:
        build_semantic_index(
            (),
            provider,
        )
    except ValueError as exc:
        assert str(exc) == (
            "Cannot build semantic index from an empty chunk collection."
        )
    else:
        raise AssertionError(
            "Expected empty corpus to raise ValueError."
        )

def test_semantic_search_rejects_empty_index():
    provider = FakeEmbeddingProvider()

    try:
        search_semantic_index(
            (),
            query="academic dress",
            embedding_provider=provider,
        )
    except ValueError as exc:
        assert str(exc) == (
            "Cannot search an empty semantic index."
        )
    else:
        raise AssertionError(
            "Expected empty index to raise ValueError."
        )

def test_cosine_similarity_rejects_mismatched_dimensions():
    try:
        cosine_similarity(
            (1.0, 2.0, 3.0),
            (1.0, 2.0),
        )
    except ValueError as exc:
        assert str(exc) == (
            "Embedding dimensions must match."
        )
    else:
        raise AssertionError(
            "Expected mismatched dimensions to raise ValueError."
        )