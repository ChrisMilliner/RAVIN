from typing import cast
import pytest
from backend.retrieval.context import (
    GroundedContext,
)
from backend.retrieval.production import (
    GroundedRetrievalResult,
)
from backend.routing.evidence_assessor import (
    assess_evidence_sufficiency,
)
from backend.routing.models import (
    EvidenceAssessment,
    EvidenceSignals,
    EvidenceSufficiency,
    QuestionIntent,
)

def _make_retrieval_result(
) -> GroundedRetrievalResult:
    return GroundedRetrievalResult(
        retrieval_results=(),
        context_chunks=(),
        context=GroundedContext(
            blocks=(),
        ),
        rendered_context="",
    )

def _make_signals() -> EvidenceSignals:
    return EvidenceSignals(
        retrieved_count=0,
        context_block_count=0,
        distinct_policy_count=0,
        top_score=None,
        second_score=None,
        score_margin=None,
    )

class FakeAssessor:
    def __init__(
        self,
        sufficiency: EvidenceSufficiency,
    ) -> None:
        self.sufficiency = sufficiency
        self.received_question: str | None = None
        self.received_intent: (
            QuestionIntent | None
        ) = None
        self.received_result: (
            GroundedRetrievalResult | None
        ) = None

    def assess(
        self,
        question: str,
        intent: QuestionIntent,
        retrieval_result: (
            GroundedRetrievalResult
        ),
    ) -> EvidenceAssessment:
        self.received_question = question
        self.received_intent = intent
        self.received_result = retrieval_result

        return EvidenceAssessment(
            sufficiency=self.sufficiency,
            signals=_make_signals(),
            reason="Fake evidence assessment.",
        )

def test_assesses_focused_question():
    retrieval_result = (
        _make_retrieval_result()
    )

    assessor = FakeAssessor(
        EvidenceSufficiency.SUFFICIENT
    )

    assessment = assess_evidence_sufficiency(
        "  What does the policy say?  ",
        QuestionIntent.FOCUSED,
        retrieval_result,
        assessor,
    )

    assert assessment.sufficiency is (
        EvidenceSufficiency.SUFFICIENT
    )

    assert assessor.received_question == (
        "What does the policy say?"
    )

    assert assessor.received_intent is (
        QuestionIntent.FOCUSED
    )

    assert assessor.received_result is (
        retrieval_result
    )

def test_assesses_broad_question():
    assessor = FakeAssessor(
        EvidenceSufficiency.INSUFFICIENT
    )

    assessment = assess_evidence_sufficiency(
        "What support is available "
        "throughout academic progression?",
        QuestionIntent.BROAD,
        _make_retrieval_result(),
        assessor,
    )

    assert assessment.sufficiency is (
        EvidenceSufficiency.INSUFFICIENT
    )

    assert assessor.received_intent is (
        QuestionIntent.BROAD
    )

def test_rejects_empty_question():
    assessor = FakeAssessor(
        EvidenceSufficiency.SUFFICIENT
    )

    with pytest.raises(
        ValueError,
        match="Question cannot be empty.",
    ):
        assess_evidence_sufficiency(
            "   ",
            QuestionIntent.FOCUSED,
            _make_retrieval_result(),
            assessor,
        )

def test_rejects_invalid_intent():
    invalid_intent = cast(
        QuestionIntent,
        "focused",
    )

    assessor = FakeAssessor(
        EvidenceSufficiency.SUFFICIENT
    )

    with pytest.raises(
        ValueError,
        match=(
            "Question intent must be a "
            "QuestionIntent."
        ),
    ):
        assess_evidence_sufficiency(
            "Question?",
            invalid_intent,
            _make_retrieval_result(),
            assessor,
        )

def test_rejects_ambiguous_question():
    assessor = FakeAssessor(
        EvidenceSufficiency.SUFFICIENT
    )

    with pytest.raises(
        ValueError,
        match=(
            "Evidence sufficiency must not be "
            "assessed for an ambiguous question."
        ),
    ):
        assess_evidence_sufficiency(
            "What do I need to submit?",
            QuestionIntent.AMBIGUOUS,
            _make_retrieval_result(),
            assessor,
        )

def test_rejects_invalid_retrieval_result():
    invalid_result = cast(
        GroundedRetrievalResult,
        "invalid",
    )

    assessor = FakeAssessor(
        EvidenceSufficiency.SUFFICIENT
    )

    with pytest.raises(
        ValueError,
        match=(
            "Retrieval result must be a "
            "GroundedRetrievalResult."
        ),
    ):
        assess_evidence_sufficiency(
            "Question?",
            QuestionIntent.FOCUSED,
            invalid_result,
            assessor,
        )

def test_rejects_invalid_assessor_result():
    class InvalidAssessor:
        def assess(
            self,
            question: str,
            intent: QuestionIntent,
            retrieval_result: (
                GroundedRetrievalResult
            ),
        ) -> EvidenceAssessment:
            return cast(
                EvidenceAssessment,
                "invalid",
            )

    with pytest.raises(
        ValueError,
        match=(
            "Evidence assessor must return an "
            "EvidenceAssessment."
        ),
    ):
        assess_evidence_sufficiency(
            "Question?",
            QuestionIntent.FOCUSED,
            _make_retrieval_result(),
            InvalidAssessor(),
        )