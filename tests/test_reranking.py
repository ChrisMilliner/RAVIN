import pytest

from backend.ingestion.models import PolicyChunk
from backend.retrieval.models import RetrievalResult
from backend.retrieval.reranking import (
    rerank_results,
)

class FakeRerankerProvider:
    def score(
        self,
        query: str,
        documents: tuple[str, ...],
    ) -> tuple[float, ...]:
        return tuple(
            0.9
            if "definition" in document.casefold()
            else 0.4
            for document in documents
        )

class WrongLengthRerankerProvider:
    def score(
        self,
        query: str,
        documents: tuple[str, ...],
    ) -> tuple[float, ...]:
        return (0.5,)

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
            0.5
            for _ in documents
        )

def make_chunk(
    policy_id: str,
    title: str,
    text: str,
    heading_path: tuple[str, ...],
    chunk_index: int = 0,
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
        chunk_index=chunk_index,
        text=text,
        heading_path=heading_path,
    )

def test_reranker_reorders_existing_results():
    policy_statement = make_chunk(
        policy_id="420",
        title="Academic Staff Qualifications Policy",
        text=(
            "Professional Equivalence is applied "
            "to some teaching staff."
        ),
        heading_path=(
            "Section 5 - Policy Statement",
        ),
    )

    definition = make_chunk(
        policy_id="420",
        title="Academic Staff Qualifications Policy",
        text=(
            "Definition: Professional Equivalence "
            "is a status conferred on an individual."
        ),
        heading_path=(
            "Section 7 - Definitions",
        ),
        chunk_index=1,
    )

    results = (
        RetrievalResult(
            chunk=policy_statement,
            score=0.90,
        ),
        RetrievalResult(
            chunk=definition,
            score=0.70,
        ),
    )

    reranked = rerank_results(
        query=(
            "What does Professional "
            "Equivalence mean?"
        ),
        results=results,
        reranker_provider=(
            FakeRerankerProvider()
        ),
    )

    assert reranked[0].chunk is definition
    assert reranked[1].chunk is policy_statement

    assert reranked[0].score == pytest.approx(
        0.9
    )

    assert reranked[1].score == pytest.approx(
        0.4
    )

def test_reranker_preserves_candidate_set():
    first = make_chunk(
        policy_id="169",
        title="Admissions Policy",
        text="Admissions policy content.",
        heading_path=("Section 5",),
    )

    second = make_chunk(
        policy_id="340",
        title="Admissions Procedure",
        text="Admissions procedure content.",
        heading_path=("Section 6",),
        chunk_index=1,
    )

    results = (
        RetrievalResult(
            chunk=first,
            score=0.8,
        ),
        RetrievalResult(
            chunk=second,
            score=0.7,
        ),
    )

    reranked = rerank_results(
        query="What are the admission requirements?",
        results=results,
        reranker_provider=(
            CapturingRerankerProvider()
        ),
    )

    assert len(reranked) == len(results)

    assert {
        result.chunk
        for result in reranked
    } == {
        result.chunk
        for result in results
    }

def test_reranker_uses_full_retrieval_text():
    chunk = make_chunk(
        policy_id="420",
        title="Academic Staff Qualifications Policy",
        text="Definition content.",
        heading_path=(
            "Section 7 - Definitions",
        ),
    )

    provider = CapturingRerankerProvider()

    rerank_results(
        query="What does this term mean?",
        results=(
            RetrievalResult(
                chunk=chunk,
                score=0.5,
            ),
        ),
        reranker_provider=provider,
    )

    assert provider.query == (
        "What does this term mean?"
    )

    assert len(provider.documents) == 1

    document = provider.documents[0]

    assert (
        "Academic Staff Qualifications Policy"
        in document
    )

    assert (
        "Section 7 - Definitions"
        in document
    )

    assert "Definition content." in document

def test_reranker_rejects_empty_query():
    chunk = make_chunk(
        policy_id="420",
        title="Policy",
        text="Content.",
        heading_path=("Section 1",),
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty.",
    ):
        rerank_results(
            query="   ",
            results=(
                RetrievalResult(
                    chunk=chunk,
                    score=0.5,
                ),
            ),
            reranker_provider=(
                FakeRerankerProvider()
            ),
        )

def test_reranker_rejects_empty_results():
    with pytest.raises(
        ValueError,
        match=(
            "Cannot rerank an empty result set."
        ),
    ):
        rerank_results(
            query="Example question",
            results=(),
            reranker_provider=(
                FakeRerankerProvider()
            ),
        )

def test_reranker_rejects_wrong_score_count():
    first = make_chunk(
        policy_id="169",
        title="Admissions Policy",
        text="First.",
        heading_path=("Section 1",),
    )

    second = make_chunk(
        policy_id="340",
        title="Admissions Procedure",
        text="Second.",
        heading_path=("Section 1",),
        chunk_index=1,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Reranker returned an unexpected "
            "number of scores."
        ),
    ):
        rerank_results(
            query="Admissions question",
            results=(
                RetrievalResult(
                    chunk=first,
                    score=0.8,
                ),
                RetrievalResult(
                    chunk=second,
                    score=0.7,
                ),
            ),
            reranker_provider=(
                WrongLengthRerankerProvider()
            ),
        )