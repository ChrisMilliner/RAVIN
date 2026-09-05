from typing import cast
from backend.retrieval.production import (
    GroundedRetrievalResult,
)
import backend.routing.routing_orchestrator as orchestrator_module
from backend.behavior import (
    AnswerBehavior,
)
from backend.routing.models import (
    EvidenceAssessment,
    EvidenceSignals,
    EvidenceSufficiency,
    QuestionIntent,
)
from backend.routing.routing_orchestrator import (
    orchestrate_answer_routing,
)

class FakeIntentClassifier:
    def __init__(
        self,
        intent: QuestionIntent,
    ) -> None:
        self._intent = intent

    def classify(
        self,
        question: str,
    ) -> QuestionIntent:
        return self._intent

class UnusedEvidenceAssessor:
    def assess(
        self,
        question,
        intent,
        retrieval_result,
    ):
        raise AssertionError(
            "Evidence assessor should not "
            "have been called."
        )

class FakeEvidenceAssessor:
    def assess(
        self,
        question,
        intent,
        retrieval_result,
    ):
        raise AssertionError(
            "The wrapper is monkeypatched "
            "in these tests."
        )

def _retrieval_result(
) -> GroundedRetrievalResult:
    return cast(
        GroundedRetrievalResult,
        object(),
    )

def _signals() -> EvidenceSignals:
    return EvidenceSignals(
        retrieved_count=1,
        context_block_count=1,
        distinct_policy_count=1,
        top_score=1.0,
        second_score=None,
        score_margin=None,
    )

def _assessment(
    sufficiency: EvidenceSufficiency,
) -> EvidenceAssessment:
    return EvidenceAssessment(
        sufficiency=sufficiency,
        signals=_signals(),
        reason="Test evidence assessment.",
    )

def test_ambiguous_question_routes_to_clarify_without_retrieval():
    result = orchestrate_answer_routing(
        "What about the policy?",
        retrieval_result=None,
        intent_classifier=(
            FakeIntentClassifier(
                QuestionIntent.AMBIGUOUS
            )
        ),
        evidence_assessor=(
            UnusedEvidenceAssessor()
        ),
    )

    assert (
        result.behavior
        == AnswerBehavior.CLARIFY
    )

    assert (
        result.evidence_assessment.sufficiency
        == EvidenceSufficiency.UNCERTAIN
    )

def test_focused_sufficient_question_routes_to_direct_answer(
    monkeypatch,
):
    monkeypatch.setattr(
        orchestrator_module,
        "assess_evidence_sufficiency",
        lambda *args, **kwargs: (
            _assessment(
                EvidenceSufficiency.SUFFICIENT
            )
        ),
    )

    result = orchestrate_answer_routing(
        "What does the policy require?",
        retrieval_result=_retrieval_result(),
        intent_classifier=(
            FakeIntentClassifier(
                QuestionIntent.FOCUSED
            )
        ),
        evidence_assessor=(
            FakeEvidenceAssessor()
        ),
    )

    assert (
        result.behavior
        == AnswerBehavior.DIRECT_ANSWER
    )

def test_broad_sufficient_question_routes_to_grounded_overview(
    monkeypatch,
):
    monkeypatch.setattr(
        orchestrator_module,
        "assess_evidence_sufficiency",
        lambda *args, **kwargs: (
            _assessment(
                EvidenceSufficiency.SUFFICIENT
            )
        ),
    )

    result = orchestrate_answer_routing(
        "Summarise the policy.",
        retrieval_result=_retrieval_result(),
        intent_classifier=(
            FakeIntentClassifier(
                QuestionIntent.BROAD
            )
        ),
        evidence_assessor=(
            FakeEvidenceAssessor()
        ),
    )

    assert (
        result.behavior
        == AnswerBehavior.GROUNDED_OVERVIEW
    )

def test_clear_insufficient_question_routes_to_no_grounded_answer(
    monkeypatch,
):
    monkeypatch.setattr(
        orchestrator_module,
        "assess_evidence_sufficiency",
        lambda *args, **kwargs: (
            _assessment(
                EvidenceSufficiency.INSUFFICIENT
            )
        ),
    )

    result = orchestrate_answer_routing(
        "What does the policy require?",
        retrieval_result=_retrieval_result(),
        intent_classifier=(
            FakeIntentClassifier(
                QuestionIntent.FOCUSED
            )
        ),
        evidence_assessor=(
            FakeEvidenceAssessor()
        ),
    )

    assert (
        result.behavior
        == AnswerBehavior.NO_GROUNDED_ANSWER
    )

def test_clear_uncertain_question_routes_to_no_grounded_answer(
    monkeypatch,
):
    monkeypatch.setattr(
        orchestrator_module,
        "assess_evidence_sufficiency",
        lambda *args, **kwargs: (
            _assessment(
                EvidenceSufficiency.UNCERTAIN
            )
        ),
    )

    result = orchestrate_answer_routing(
        "What does the policy require?",
        retrieval_result=_retrieval_result(),
        intent_classifier=(
            FakeIntentClassifier(
                QuestionIntent.FOCUSED
            )
        ),
        evidence_assessor=(
            FakeEvidenceAssessor()
        ),
    )

    assert (
        result.behavior
        == AnswerBehavior.NO_GROUNDED_ANSWER
    )

def test_clear_question_requires_retrieval_result():
    try:
        orchestrate_answer_routing(
            "What does the policy require?",
            retrieval_result=None,
            intent_classifier=(
                FakeIntentClassifier(
                    QuestionIntent.FOCUSED
                )
            ),
            evidence_assessor=(
                FakeEvidenceAssessor()
            ),
        )

    except ValueError as error:
        assert str(error) == (
            "A retrieval result is required "
            "for a clear question."
        )

    else:
        raise AssertionError(
            "Expected ValueError."
        )