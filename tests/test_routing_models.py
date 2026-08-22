import pytest
from backend.behavior import AnswerBehavior
from backend.routing.models import (
    EvidenceSignals,
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

def test_routing_result_preserves_behavior_and_evidence():
    evidence = EvidenceSignals(
        retrieved_count=5,
        context_block_count=2,
        distinct_policy_count=1,
        top_score=7.0,
        second_score=5.0,
        score_margin=2.0,
    )

    result = RoutingResult(
        behavior=AnswerBehavior.DIRECT_ANSWER,
        evidence=evidence,
        reason="Evidence supports a focused answer.",
    )

    assert result.behavior is (
        AnswerBehavior.DIRECT_ANSWER
    )
    assert result.evidence is evidence
    assert result.reason == (
        "Evidence supports a focused answer."
    )

def test_routing_result_rejects_empty_reason():
    evidence = EvidenceSignals(
        retrieved_count=0,
        context_block_count=0,
        distinct_policy_count=0,
        top_score=None,
        second_score=None,
        score_margin=None,
    )

    with pytest.raises(
        ValueError,
        match="Routing reason cannot be empty.",
    ):
        RoutingResult(
            behavior=(
                AnswerBehavior.NO_GROUNDED_ANSWER
            ),
            evidence=evidence,
            reason="   ",
        )