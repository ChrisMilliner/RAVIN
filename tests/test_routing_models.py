import pytest
from typing import cast
from backend.behavior import AnswerBehavior
from backend.routing.models import (
    EvidenceAssessment,
    EvidenceSignals,
    EvidenceSufficiency,
    QuestionAssessment,
    QuestionIntent,
    RoutingResult,
)

def test_evidence_signals_preserve_values():
    signals = EvidenceSignals(
        retrieved_count=5,
        context_block_count=4,
        distinct_policy_count=2,
        top_score=7.5,
        second_score=6.0,
        score_margin=1.5,
    )

    assert signals.retrieved_count == 5
    assert signals.context_block_count == 4
    assert signals.distinct_policy_count == 2
    assert signals.top_score == 7.5
    assert signals.second_score == 6.0
    assert signals.score_margin == 1.5

def test_evidence_signals_reports_evidence_present():
    signals = EvidenceSignals(
        retrieved_count=5,
        context_block_count=2,
        distinct_policy_count=1,
        top_score=5.0,
        second_score=4.0,
        score_margin=1.0,
    )

    assert signals.has_evidence is True

def test_empty_evidence_signals_reports_no_evidence():
    signals = EvidenceSignals(
        retrieved_count=0,
        context_block_count=0,
        distinct_policy_count=0,
        top_score=None,
        second_score=None,
        score_margin=None,
    )

    assert signals.has_evidence is False

def test_empty_retrieval_rejects_scores():
    with pytest.raises(
        ValueError,
        match=(
            "Empty retrieval cannot define "
            "retrieval scores."
        ),
    ):
        EvidenceSignals(
            retrieved_count=0,
            context_block_count=0,
            distinct_policy_count=0,
            top_score=1.0,
            second_score=None,
            score_margin=None,
        )

def test_single_result_requires_top_score():
    with pytest.raises(
        ValueError,
        match=(
            "Single-result retrieval must "
            "define a top score."
        ),
    ):
        EvidenceSignals(
            retrieved_count=1,
            context_block_count=1,
            distinct_policy_count=1,
            top_score=None,
            second_score=None,
            score_margin=None,
        )

def test_single_result_rejects_second_score():
    with pytest.raises(
        ValueError,
        match=(
            "Single-result retrieval cannot "
            "define a second score or margin."
        ),
    ):
        EvidenceSignals(
            retrieved_count=1,
            context_block_count=1,
            distinct_policy_count=1,
            top_score=2.0,
            second_score=1.0,
            score_margin=1.0,
        )

def test_multi_result_requires_all_score_signals():
    with pytest.raises(
        ValueError,
        match=(
            "Multi-result retrieval must "
            "define top, second and margin "
            "scores."
        ),
    ):
        EvidenceSignals(
            retrieved_count=2,
            context_block_count=1,
            distinct_policy_count=1,
            top_score=2.0,
            second_score=None,
            score_margin=None,
        )

def test_distinct_policy_count_cannot_exceed_blocks():
    with pytest.raises(
        ValueError,
        match=(
            "Distinct policy count cannot exceed "
            "the context block count."
        ),
    ):
        EvidenceSignals(
            retrieved_count=2,
            context_block_count=1,
            distinct_policy_count=2,
            top_score=2.0,
            second_score=1.0,
            score_margin=1.0,
        )

def test_routing_result_preserves_assessments():
    signals = EvidenceSignals(
        retrieved_count=5,
        context_block_count=2,
        distinct_policy_count=1,
        top_score=7.0,
        second_score=5.0,
        score_margin=2.0,
    )

    question_assessment = QuestionAssessment(
        intent=QuestionIntent.FOCUSED,
        reason="The question asks for one fact.",
    )

    evidence_assessment = EvidenceAssessment(
        sufficiency=(
            EvidenceSufficiency.SUFFICIENT
        ),
        signals=signals,
        reason=(
            "Retrieved evidence directly "
            "addresses the question."
        ),
    )

    result = RoutingResult(
        behavior=AnswerBehavior.DIRECT_ANSWER,
        question_assessment=(
            question_assessment
        ),
        evidence_assessment=(
            evidence_assessment
        ),
        reason=(
            "Focused question with sufficient "
            "grounded evidence."
        ),
    )

    assert result.behavior is (
        AnswerBehavior.DIRECT_ANSWER
    )

    assert result.question_assessment is (
        question_assessment
    )

    assert result.evidence_assessment is (
        evidence_assessment
    )

    assert result.reason == (
        "Focused question with sufficient "
        "grounded evidence."
    )

def test_routing_result_rejects_empty_reason():
    signals = EvidenceSignals(
        retrieved_count=0,
        context_block_count=0,
        distinct_policy_count=0,
        top_score=None,
        second_score=None,
        score_margin=None,
    )

    question_assessment = QuestionAssessment(
        intent=QuestionIntent.FOCUSED,
        reason="The question asks for one fact.",
    )

    evidence_assessment = EvidenceAssessment(
        sufficiency=(
            EvidenceSufficiency.INSUFFICIENT
        ),
        signals=signals,
        reason="No grounded evidence was found.",
    )

    with pytest.raises(
        ValueError,
        match="Routing reason cannot be empty.",
    ):
        RoutingResult(
            behavior=(
                AnswerBehavior.NO_GROUNDED_ANSWER
            ),
            question_assessment=(
                question_assessment
            ),
            evidence_assessment=(
                evidence_assessment
            ),
            reason="   ",
        )

def test_question_intent_defines_expected_values():
    assert tuple(
        intent.value
        for intent in QuestionIntent
    ) == (
        "focused",
        "broad",
        "ambiguous",
    )

def test_evidence_sufficiency_defines_expected_values():
    assert tuple(
        sufficiency.value
        for sufficiency in EvidenceSufficiency
    ) == (
        "sufficient",
        "insufficient",
        "uncertain",
    )

def test_question_assessment_preserves_values():
    assessment = QuestionAssessment(
        intent=QuestionIntent.BROAD,
        reason=(
            "The question asks about a process "
            "rather than one specific fact."
        ),
    )

    assert assessment.intent is (
        QuestionIntent.BROAD
    )

    assert assessment.reason == (
        "The question asks about a process "
        "rather than one specific fact."
    )

def test_question_assessment_rejects_invalid_intent():
    invalid_intent = cast(
        QuestionIntent,
        "invalid",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Question intent must be a "
            "QuestionIntent."
        ),
    ):
        QuestionAssessment(
            intent=invalid_intent,
            reason="Test reason.",
        )

def test_question_assessment_rejects_empty_reason():
    with pytest.raises(
        ValueError,
        match=(
            "Question assessment reason "
            "cannot be empty."
        ),
    ):
        QuestionAssessment(
            intent=QuestionIntent.FOCUSED,
            reason="   ",
        )

def test_evidence_assessment_preserves_values():
    signals = EvidenceSignals(
        retrieved_count=5,
        context_block_count=3,
        distinct_policy_count=1,
        top_score=7.0,
        second_score=5.0,
        score_margin=2.0,
    )

    assessment = EvidenceAssessment(
        sufficiency=(
            EvidenceSufficiency.SUFFICIENT
        ),
        signals=signals,
        reason=(
            "Retrieved evidence directly "
            "addresses the question."
        ),
    )

    assert assessment.sufficiency is (
        EvidenceSufficiency.SUFFICIENT
    )

    assert assessment.signals is signals

    assert assessment.reason == (
        "Retrieved evidence directly "
        "addresses the question."
    )

def test_evidence_assessment_rejects_invalid_sufficiency():
    invalid_sufficiency = cast(
        EvidenceSufficiency,
        "invalid",
    )

    signals = EvidenceSignals(
        retrieved_count=0,
        context_block_count=0,
        distinct_policy_count=0,
        top_score=None,
        second_score=None,
        score_margin=None,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Evidence sufficiency must be an "
            "EvidenceSufficiency."
        ),
    ):
        EvidenceAssessment(
            sufficiency=invalid_sufficiency,
            signals=signals,
            reason="Test reason.",
        )

def test_evidence_assessment_rejects_invalid_signals():
    invalid_signals = cast(
        EvidenceSignals,
        object(),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Evidence assessment signals must "
            "be EvidenceSignals."
        ),
    ):
        EvidenceAssessment(
            sufficiency=(
                EvidenceSufficiency.UNCERTAIN
            ),
            signals=invalid_signals,
            reason="Test reason.",
        )

def test_evidence_assessment_rejects_empty_reason():
    signals = EvidenceSignals(
        retrieved_count=0,
        context_block_count=0,
        distinct_policy_count=0,
        top_score=None,
        second_score=None,
        score_margin=None,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Evidence assessment reason "
            "cannot be empty."
        ),
    ):
        EvidenceAssessment(
            sufficiency=(
                EvidenceSufficiency.UNCERTAIN
            ),
            signals=signals,
            reason="   ",
        )

def test_routing_result_rejects_invalid_question_assessment():
    invalid_question_assessment = cast(
        QuestionAssessment,
        object(),
    )

    signals = EvidenceSignals(
        retrieved_count=0,
        context_block_count=0,
        distinct_policy_count=0,
        top_score=None,
        second_score=None,
        score_margin=None,
    )

    evidence_assessment = EvidenceAssessment(
        sufficiency=(
            EvidenceSufficiency.INSUFFICIENT
        ),
        signals=signals,
        reason="No grounded evidence was found.",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Routing question assessment must "
            "be QuestionAssessment."
        ),
    ):
        RoutingResult(
            behavior=(
                AnswerBehavior.NO_GROUNDED_ANSWER
            ),
            question_assessment=(
                invalid_question_assessment
            ),
            evidence_assessment=(
                evidence_assessment
            ),
            reason="Test reason.",
        )

def test_routing_result_rejects_invalid_evidence_assessment():
    question_assessment = QuestionAssessment(
        intent=QuestionIntent.FOCUSED,
        reason="The question asks for one fact.",
    )

    invalid_evidence_assessment = cast(
        EvidenceAssessment,
        object(),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Routing evidence assessment must "
            "be EvidenceAssessment."
        ),
    ):
        RoutingResult(
            behavior=AnswerBehavior.DIRECT_ANSWER,
            question_assessment=(
                question_assessment
            ),
            evidence_assessment=(
                invalid_evidence_assessment
            ),
            reason="Test reason.",
        )

def test_ambiguous_question_preserves_clarification_options():
    assessment = QuestionAssessment(
        intent=QuestionIntent.AMBIGUOUS,
        reason=(
            "The question could refer to more "
            "than one university process."
        ),
        clarification_options=(
            (
                "What do I need to submit for an "
                "academic progression review?"
            ),
            (
                "What documents do I need to "
                "submit for admission?"
            ),
        ),
    )

    assert assessment.clarification_options == (
        (
            "What do I need to submit for an "
            "academic progression review?"
        ),
        (
            "What documents do I need to "
            "submit for admission?"
        ),
    )

def test_non_ambiguous_question_rejects_clarification_options():
    with pytest.raises(
        ValueError,
        match=(
            "Clarification options are only "
            "valid for ambiguous questions."
        ),
    ):
        QuestionAssessment(
            intent=QuestionIntent.FOCUSED,
            reason=(
                "The question asks for one "
                "specific fact."
            ),
            clarification_options=(
                "Did you mean something else?",
            ),
        )

def test_question_assessment_rejects_empty_clarification_option():
    with pytest.raises(
        ValueError,
        match=(
            "Clarification options cannot "
            "contain empty values."
        ),
    ):
        QuestionAssessment(
            intent=QuestionIntent.AMBIGUOUS,
            reason="The question is unclear.",
            clarification_options=(
                "Valid suggested question?",
                "   ",
            ),
        )