from backend.evaluation.experiment_models import (
    DatasetValidationStatus,
    ExperimentDirection,
    ExperimentSelectionDecision,
    MetricComparison,
    QuestionRankChange,
    RetrievalExperimentComparison,
    RetrievalExperimentConfig,
)
from backend.evaluation.models import EvaluationRunResult

def _compare_direction(
    baseline: EvaluationRunResult,
    candidate: EvaluationRunResult,
) -> ExperimentDirection:
    if candidate.top_1_accuracy > baseline.top_1_accuracy:
        return ExperimentDirection.IMPROVED

    if candidate.top_1_accuracy < baseline.top_1_accuracy:
        return ExperimentDirection.REGRESSED

    if candidate.mrr > baseline.mrr:
        return ExperimentDirection.IMPROVED

    if candidate.mrr < baseline.mrr:
        return ExperimentDirection.REGRESSED

    if candidate.hit_at_k > baseline.hit_at_k:
        return ExperimentDirection.IMPROVED

    if candidate.hit_at_k < baseline.hit_at_k:
        return ExperimentDirection.REGRESSED

    return ExperimentDirection.UNCHANGED

def _selection_decision(
    direction: ExperimentDirection,
    quality_gate_passed: bool,
    dataset_status: DatasetValidationStatus,
) -> ExperimentSelectionDecision:
    if not quality_gate_passed:
        return (
            ExperimentSelectionDecision
            .REJECT_BELOW_THRESHOLD
        )

    if (
        dataset_status
        != DatasetValidationStatus.HUMAN_VALIDATED
    ):
        return (
            ExperimentSelectionDecision
            .REQUIRES_VALIDATED_EVALUATION
        )

    if direction != ExperimentDirection.IMPROVED:
        return (
            ExperimentSelectionDecision
            .REJECT_NOT_IMPROVED
        )

    return (
        ExperimentSelectionDecision
        .ELIGIBLE_FOR_SELECTION
    )

def compare_retrieval_experiments(
    baseline: EvaluationRunResult,
    candidate: EvaluationRunResult,
    config: RetrievalExperimentConfig,
) -> RetrievalExperimentComparison:
    if baseline.top_k != candidate.top_k:
        raise ValueError(
            "Baseline and candidate must use the same top_k."
        )

    if baseline.top_k != config.top_k:
        raise ValueError(
            "Experiment configuration top_k must match evaluation results."
        )

    baseline_by_id = {
        result.question_id: result
        for result in baseline.question_results
    }

    candidate_by_id = {
        result.question_id: result
        for result in candidate.question_results
    }

    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError(
            "Baseline and candidate must evaluate the same questions."
        )

    question_rank_changes = tuple(
        QuestionRankChange(
            question_id=question_id,
            baseline_rank=(
                baseline_by_id[
                    question_id
                ].first_relevant_rank
            ),
            candidate_rank=(
                candidate_by_id[
                    question_id
                ].first_relevant_rank
            ),
        )
        for question_id in baseline_by_id
    )

    quality_gate_passed = (
        candidate.top_1_accuracy
        >= config.quality_threshold
    )

    validated_dataset_gate_passed = (
        quality_gate_passed
        and config.dataset_status
        == DatasetValidationStatus.HUMAN_VALIDATED
    )

    direction = _compare_direction(
        baseline,
        candidate,
    )

    selection_decision = _selection_decision(
        direction=direction,
        quality_gate_passed=quality_gate_passed,
        dataset_status=config.dataset_status,
    )

    return RetrievalExperimentComparison(
        config=config,
        top_1=MetricComparison(
            baseline=baseline.top_1_accuracy,
            candidate=candidate.top_1_accuracy,
        ),
        hit_at_k=MetricComparison(
            baseline=baseline.hit_at_k,
            candidate=candidate.hit_at_k,
        ),
        mrr=MetricComparison(
            baseline=baseline.mrr,
            candidate=candidate.mrr,
        ),
        question_rank_changes=(
            question_rank_changes
        ),
        direction=direction,
        selection_decision=(
            selection_decision
        ),
        quality_gate_passed=(
            quality_gate_passed
        ),
        validated_dataset_gate_passed=(
            validated_dataset_gate_passed
        ),
    )