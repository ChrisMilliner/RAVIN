import pytest
from backend.ingestion.models import PolicyChunk
from backend.retrieval.hybrid import (
    calculate_hybrid_score,
    calculate_lexical_coverage,
    search_hybrid_index,
    tokenize_lexical_text,
)
from backend.retrieval.index import build_semantic_index

class FixedEmbeddingProvider:
    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            (1.0, 0.0)
            for _ in texts
        )

    def embed_query(
        self,
        text: str,
    ) -> tuple[float, ...]:
        return (1.0, 0.0)

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

def test_tokenize_lexical_text_normalizes_and_removes_stop_words():
    tokens = tokenize_lexical_text(
        "What IS the Course-Transfer Policy?"
    )

    assert tokens == frozenset(
        {
            "course",
            "transfer",
            "policy",
        }
    )

def test_calculate_lexical_coverage_measures_query_term_overlap():
    score = calculate_lexical_coverage(
        "student transfer sanction",
        "The student submitted a course transfer application.",
    )

    assert score == pytest.approx(
        2 / 3
    )

def test_lexical_coverage_returns_zero_without_meaningful_query_tokens():
    score = calculate_lexical_coverage(
        "what is the",
        "Example policy content.",
    )

    assert score == 0.0

def test_calculate_hybrid_score_uses_default_weights():
    score = calculate_hybrid_score(
        semantic_score=0.60,
        lexical_score=0.80,
    )

    assert score == pytest.approx(
        0.63
    )

def test_hybrid_score_rejects_negative_weights():
    with pytest.raises(
        ValueError,
        match="Semantic weight cannot be negative.",
    ):
        calculate_hybrid_score(
            semantic_score=0.50,
            lexical_score=0.50,
            semantic_weight=-0.10,
            lexical_weight=1.10,
        )

    with pytest.raises(
        ValueError,
        match="Lexical weight cannot be negative.",
    ):
        calculate_hybrid_score(
            semantic_score=0.50,
            lexical_score=0.50,
            semantic_weight=1.10,
            lexical_weight=-0.10,
        )

def test_hybrid_score_requires_weights_sum_to_one():
    with pytest.raises(
        ValueError,
        match=(
            "Hybrid retrieval weights must sum to 1."
        ),
    ):
        calculate_hybrid_score(
            semantic_score=0.50,
            lexical_score=0.50,
            semantic_weight=0.80,
            lexical_weight=0.30,
        )

def test_hybrid_search_uses_lexical_signal_to_break_semantic_tie():
    provider = FixedEmbeddingProvider()

    generic_transfer_chunk = make_chunk(
        policy_id="340",
        title="Admissions Procedure",
        text=(
            "A student may submit a course "
            "transfer application."
        ),
        heading_path=(
            "Section 6 - Procedures",
            "Part F - Transfers",
        ),
    )

    sanction_chunk = make_chunk(
        policy_id="220",
        title="Academic Progression Review Policy",
        text=(
            "A student enrolment sanction remains "
            "active after a course transfer."
        ),
        heading_path=(
            "Section 6 - Procedures",
            "Course Transfers and New Courses",
        ),
    )

    index = build_semantic_index(
        (
            generic_transfer_chunk,
            sanction_chunk,
        ),
        provider,
    )

    results = search_hybrid_index(
        index,
        query="student transfer sanction",
        embedding_provider=provider,
        top_k=2,
    )

    assert results[0].chunk is sanction_chunk
    assert (
        results[0].score
        > results[1].score
    )

def test_hybrid_search_respects_top_k():
    provider = FixedEmbeddingProvider()

    chunks = (
        make_chunk(
            "220",
            "Progression Policy",
            "Student sanction requirements.",
            ("Section 1",),
        ),
        make_chunk(
            "340",
            "Admissions Procedure",
            "Student transfer requirements.",
            ("Section 1",),
        ),
    )

    index = build_semantic_index(
        chunks,
        provider,
    )

    results = search_hybrid_index(
        index,
        query="student sanction",
        embedding_provider=provider,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.policy_id == "220"

def test_hybrid_search_rejects_empty_query():
    provider = FixedEmbeddingProvider()

    chunk = make_chunk(
        "220",
        "Progression Policy",
        "Student sanction requirements.",
        ("Section 1",),
    )

    index = build_semantic_index(
        (chunk,),
        provider,
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty.",
    ):
        search_hybrid_index(
            index,
            query="   ",
            embedding_provider=provider,
        )

def test_hybrid_search_rejects_invalid_top_k():
    provider = FixedEmbeddingProvider()

    chunk = make_chunk(
        "220",
        "Progression Policy",
        "Student sanction requirements.",
        ("Section 1",),
    )

    index = build_semantic_index(
        (chunk,),
        provider,
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero.",
    ):
        search_hybrid_index(
            index,
            query="student sanction",
            embedding_provider=provider,
            top_k=0,
        )

def test_hybrid_search_rejects_empty_index():
    provider = FixedEmbeddingProvider()

    with pytest.raises(
        ValueError,
        match=(
            "Cannot search an empty semantic index."
        ),
    ):
        search_hybrid_index(
            (),
            query="student sanction",
            embedding_provider=provider,
        )