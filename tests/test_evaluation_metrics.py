import pytest
from backend.evaluation.metrics import (
    calculate_hit_at_k,
    calculate_mrr,
    calculate_top_1_accuracy,
    meets_top_1_quality_gate,
)

def test_top_1_accuracy_counts_only_rank_one_hits():
    accuracy = calculate_top_1_accuracy(
        (
            1,
            3,
            None,
            1,
        )
    )

    assert accuracy == 0.5

def test_hit_at_k_counts_results_within_k():
    hit_rate = calculate_hit_at_k(
        (
            1,
            3,
            None,
            6,
        ),
        k=5,
    )

    assert hit_rate == 0.5

def test_mrr_rewards_higher_ranked_evidence():
    mrr = calculate_mrr(
        (
            1,
            3,
            None,
        )
    )

    assert mrr == pytest.approx(
        (1.0 + (1.0 / 3.0)) / 3.0
    )

def test_perfect_results_produce_perfect_metrics():
    ranks = (
        1,
        1,
        1,
        1,
    )

    assert calculate_top_1_accuracy(ranks) == 1.0
    assert calculate_hit_at_k(ranks, k=5) == 1.0
    assert calculate_mrr(ranks) == 1.0

def test_metrics_reject_empty_evaluation_set():
    with pytest.raises(
        ValueError,
        match=(
            "Cannot calculate metrics from an empty evaluation set."
        ),
    ):
        calculate_top_1_accuracy(())

def test_hit_at_k_rejects_invalid_k():
    with pytest.raises(
        ValueError,
        match="k must be greater than zero.",
    ):
        calculate_hit_at_k(
            (1, 2),
            k=0,
        )

def test_quality_gate_passes_at_exact_threshold():
    assert meets_top_1_quality_gate(
        top_1_accuracy=0.95,
        threshold=0.95,
    )


def test_quality_gate_fails_below_threshold():
    assert not meets_top_1_quality_gate(
        top_1_accuracy=0.949,
        threshold=0.95,
    )