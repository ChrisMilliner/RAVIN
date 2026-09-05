from backend.behavior import AnswerBehavior
from backend.routing.models import (
    EvidenceAssessment,
    EvidenceSignals,
    EvidenceSufficiency,
    QuestionAssessment,
    QuestionIntent,
)
from backend.routing.router import (
    route_answer,
)

def make_evidence_assessment(
    sufficiency: EvidenceSufficiency,
) -> EvidenceAssessment:
    if (
        sufficiency
        == EvidenceSufficiency.INSUFFICIENT
    ):
        signals = EvidenceSignals(
            retrieved_count=0,
            context_block_count=0,
            distinct_policy_count=0,
            top_score=None,
            second_score=None,
            score_margin=None,
        )
    else:
        signals = EvidenceSignals(
            retrieved_count=5,
            context_block_count=3,
            distinct_policy_count=1,
            top_score=7.0,
            second_score=5.0,
            score_margin=2.0,
        )

    return EvidenceAssessment(
        sufficiency=sufficiency,
        signals=signals,
        reason=(
            "Test evidence assessment."
        ),
    )

def make_question_assessment(
    intent: QuestionIntent,
) -> QuestionAssessment:
    clarification_options = ()

    if intent == QuestionIntent.AMBIGUOUS:
        clarification_options = (
            (
                "What do I need to submit for an "
                "academic progression review?"
            ),
            (
                "What documents do I need to "
                "submit for admission?"
            ),
        )

    return QuestionAssessment(
        intent=intent,
        reason="Test question assessment.",
        clarification_options=(
            clarification_options
        ),
    )

def test_ambiguous_question_routes_to_clarify_with_evidence():
    question_assessment = (
        make_question_assessment(
            QuestionIntent.AMBIGUOUS
        )
    )

    evidence_assessment = (
        make_evidence_assessment(
            EvidenceSufficiency.SUFFICIENT
        )
    )

    result = route_answer(
        question_assessment,
        evidence_assessment,
    )

    assert result.behavior is (
        AnswerBehavior.CLARIFY
    )

    assert (
        result.question_assessment
        is question_assessment
    )

    assert (
        result.evidence_assessment
        is evidence_assessment
    )

    assert (
        result.question_assessment
        .clarification_options
    )

def test_ambiguous_question_routes_to_clarify_without_evidence():
    result = route_answer(
        make_question_assessment(
            QuestionIntent.AMBIGUOUS
        ),
        make_evidence_assessment(
            EvidenceSufficiency.INSUFFICIENT
        ),
    )

    assert result.behavior is (
        AnswerBehavior.CLARIFY
    )

def test_focused_question_with_sufficient_evidence_routes_direct():
    result = route_answer(
        make_question_assessment(
            QuestionIntent.FOCUSED
        ),
        make_evidence_assessment(
            EvidenceSufficiency.SUFFICIENT
        ),
    )

    assert result.behavior is (
        AnswerBehavior.DIRECT_ANSWER
    )

def test_broad_question_with_sufficient_evidence_routes_overview():
    result = route_answer(
        make_question_assessment(
            QuestionIntent.BROAD
        ),
        make_evidence_assessment(
            EvidenceSufficiency.SUFFICIENT
        ),
    )

    assert result.behavior is (
        AnswerBehavior.GROUNDED_OVERVIEW
    )

def test_focused_question_with_insufficient_evidence_routes_no_answer():
    result = route_answer(
        make_question_assessment(
            QuestionIntent.FOCUSED
        ),
        make_evidence_assessment(
            EvidenceSufficiency.INSUFFICIENT
        ),
    )

    assert result.behavior is (
        AnswerBehavior.NO_GROUNDED_ANSWER
    )

def test_broad_question_with_insufficient_evidence_routes_no_answer():
    result = route_answer(
        make_question_assessment(
            QuestionIntent.BROAD
        ),
        make_evidence_assessment(
            EvidenceSufficiency.INSUFFICIENT
        ),
    )

    assert result.behavior is (
        AnswerBehavior.NO_GROUNDED_ANSWER
    )

def test_focused_question_with_uncertain_evidence_fails_closed():
    result = route_answer(
        make_question_assessment(
            QuestionIntent.FOCUSED
        ),
        make_evidence_assessment(
            EvidenceSufficiency.UNCERTAIN
        ),
    )

    assert result.behavior is (
        AnswerBehavior.NO_GROUNDED_ANSWER
    )

def test_broad_question_with_uncertain_evidence_fails_closed():
    result = route_answer(
        make_question_assessment(
            QuestionIntent.BROAD
        ),
        make_evidence_assessment(
            EvidenceSufficiency.UNCERTAIN
        ),
    )

    assert result.behavior is (
        AnswerBehavior.NO_GROUNDED_ANSWER
    )