import pytest
from backend.ingestion.models import (
    PolicyChunk,
)
from backend.retrieval.context import (
    GroundedContext,
    GroundedContextBlock,
)
from backend.retrieval.models import (
    RetrievalResult,
)
from backend.retrieval.production import (
    GroundedRetrievalResult,
)
from backend.routing.signals import (
    extract_evidence_signals,
)

def make_chunk(
    policy_id: str,
    chunk_index: int,
) -> PolicyChunk:
    return PolicyChunk(
        policy_id=policy_id,
        policy_title=(
            f"Policy {policy_id}"
        ),
        source_url=(
            "https://example.invalid/"
            f"{policy_id}"
        ),
        status="Current",
        effective_date=None,
        review_date=None,
        chunk_index=chunk_index,
        text=(
            f"Evidence for policy "
            f"{policy_id}."
        ),
        heading_path=(
            "Section 5",
        ),
    )

def make_block(
    policy_id: str,
    chunk_index: int,
) -> GroundedContextBlock:
    return GroundedContextBlock(
        policy_id=policy_id,
        policy_title=(
            f"Policy {policy_id}"
        ),
        source_url=(
            "https://example.invalid/"
            f"{policy_id}"
        ),
        heading_path=(
            "Section 5",
        ),
        start_chunk_index=chunk_index,
        end_chunk_index=chunk_index,
        text=(
            f"Evidence for policy "
            f"{policy_id}."
        ),
    )

def make_result(
    retrieval_results: tuple[
        RetrievalResult,
        ...
    ],
    blocks: tuple[
        GroundedContextBlock,
        ...
    ],
) -> GroundedRetrievalResult:
    return GroundedRetrievalResult(
        retrieval_results=(
            retrieval_results
        ),
        context_chunks=tuple(
            result.chunk
            for result in retrieval_results
        ),
        context=GroundedContext(
            blocks=blocks
        ),
        rendered_context="",
    )

def test_extract_signals_handles_empty_result():
    result = make_result(
        (),
        (),
    )

    signals = extract_evidence_signals(
        result
    )

    assert signals.retrieved_count == 0
    assert signals.context_block_count == 0
    assert signals.distinct_policy_count == 0
    assert signals.top_score is None
    assert signals.second_score is None
    assert signals.score_margin is None
    assert signals.has_evidence is False

def test_extract_signals_handles_single_result():
    chunk = make_chunk(
        "220",
        7,
    )

    result = make_result(
        (
            RetrievalResult(
                chunk=chunk,
                score=4.5,
            ),
        ),
        (
            make_block(
                "220",
                7,
            ),
        ),
    )

    signals = extract_evidence_signals(
        result
    )

    assert signals.retrieved_count == 1
    assert signals.context_block_count == 1
    assert signals.distinct_policy_count == 1
    assert signals.top_score == 4.5
    assert signals.second_score is None
    assert signals.score_margin is None
    assert signals.has_evidence is True

def test_extract_signals_calculates_score_margin():
    first = make_chunk(
        "220",
        7,
    )

    second = make_chunk(
        "220",
        8,
    )

    result = make_result(
        (
            RetrievalResult(
                chunk=first,
                score=7.5,
            ),
            RetrievalResult(
                chunk=second,
                score=6.0,
            ),
        ),
        (
            make_block(
                "220",
                7,
            ),
        ),
    )

    signals = extract_evidence_signals(
        result
    )

    assert signals.top_score == 7.5
    assert signals.second_score == 6.0
    assert signals.score_margin == pytest.approx(
        1.5
    )

def test_extract_signals_handles_negative_scores():
    first = make_chunk(
        "340",
        17,
    )

    second = make_chunk(
        "220",
        5,
    )

    result = make_result(
        (
            RetrievalResult(
                chunk=first,
                score=-0.934581,
            ),
            RetrievalResult(
                chunk=second,
                score=-1.631718,
            ),
        ),
        (
            make_block(
                "340",
                17,
            ),
            make_block(
                "220",
                5,
            ),
        ),
    )

    signals = extract_evidence_signals(
        result
    )

    assert signals.top_score == pytest.approx(
        -0.934581
    )

    assert signals.second_score == pytest.approx(
        -1.631718
    )

    assert signals.score_margin == pytest.approx(
        0.697137
    )

def test_extract_signals_counts_distinct_context_policies():
    retrieval_results = (
        RetrievalResult(
            chunk=make_chunk(
                "169",
                7,
            ),
            score=6.9,
        ),
        RetrievalResult(
            chunk=make_chunk(
                "340",
                15,
            ),
            score=4.4,
        ),
    )

    blocks = (
        make_block(
            "169",
            7,
        ),
        make_block(
            "169",
            8,
        ),
        make_block(
            "340",
            15,
        ),
    )

    signals = extract_evidence_signals(
        make_result(
            retrieval_results,
            blocks,
        )
    )

    assert signals.context_block_count == 3

    assert signals.distinct_policy_count == 2