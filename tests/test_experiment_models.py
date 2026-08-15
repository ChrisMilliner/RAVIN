import pytest
from backend.evaluation.experiment_models import (
    DatasetValidationStatus,
    ExperimentSelectionDecision,
    MetricComparison,
    QuestionRankChange,
    RetrievalExperimentConfig,
)

def test_experiment_config_preserves_named_configurations():
    config = RetrievalExperimentConfig(
        experiment_name="Hybrid retrieval comparison",
        baseline_name="MiniLM semantic baseline",
        candidate_name="Hybrid semantic lexical v1",
        dataset_name="Retrieval gold standard",
        dataset_status=(
            DatasetValidationStatus.HUMAN_VALIDATED
        ),
    )

    assert config.experiment_name == (
        "Hybrid retrieval comparison"
    )
    assert config.baseline_name == (
        "MiniLM semantic baseline"
    )
    assert config.candidate_name == (
        "Hybrid semantic lexical v1"
    )

def test_experiment_config_uses_95_percent_quality_gate():
    config = RetrievalExperimentConfig(
        experiment_name="Test experiment",
        baseline_name="Baseline",
        candidate_name="Candidate",
        dataset_name="Dataset",
        dataset_status=(
            DatasetValidationStatus.PRELIMINARY
        ),
    )

    assert config.quality_threshold == 0.95
    assert config.top_k == 5

def test_dataset_status_distinguishes_preliminary_from_validated():
    assert (
        DatasetValidationStatus.PRELIMINARY
        != DatasetValidationStatus.HUMAN_VALIDATED
    )

def test_selection_decision_defines_accuracy_gate_outcomes():
    assert (
        ExperimentSelectionDecision.REJECT_BELOW_THRESHOLD.value
        == "reject-below-threshold"
    )

    assert (
        ExperimentSelectionDecision.REQUIRES_VALIDATED_EVALUATION.value
        == "requires-validated-evaluation"
    )

    assert (
        ExperimentSelectionDecision.REJECT_NOT_IMPROVED.value
        == "reject-not-improved"
    )

    assert (
        ExperimentSelectionDecision.ELIGIBLE_FOR_SELECTION.value
        == "eligible-for-selection"
    )

def test_metric_comparison_calculates_positive_delta():
    comparison = MetricComparison(
        baseline=0.80,
        candidate=0.96,
    )

    assert comparison.delta == pytest.approx(
        0.16
    )

def test_metric_comparison_calculates_negative_delta():
    comparison = MetricComparison(
        baseline=0.96,
        candidate=0.90,
    )

    assert comparison.delta == pytest.approx(
        -0.06
    )

def test_question_rank_change_supports_missing_result():
    change = QuestionRankChange(
        question_id="Q001",
        baseline_rank=None,
        candidate_rank=2,
    )

    assert change.baseline_rank is None
    assert change.candidate_rank == 2

def test_question_rank_change_rejects_invalid_rank():
    with pytest.raises(
        ValueError,
        match="Candidate rank must be greater than zero.",
    ):
        QuestionRankChange(
            question_id="Q001",
            baseline_rank=1,
            candidate_rank=0,
        )

def test_experiment_config_rejects_invalid_threshold():
    with pytest.raises(
        ValueError,
        match=(
            "Experiment quality threshold must be between 0 and 1."
        ),
    ):
        RetrievalExperimentConfig(
            experiment_name="Test experiment",
            baseline_name="Baseline",
            candidate_name="Candidate",
            dataset_name="Dataset",
            dataset_status=(
                DatasetValidationStatus.PRELIMINARY
            ),
            quality_threshold=1.01,
        )