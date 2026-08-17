import pytest
from backend.evaluation.experiment_models import (
    DatasetValidationStatus,
    ExperimentDirection,
    ExperimentSelectionDecision,
    RetrievalExperimentConfig,
)
from backend.evaluation.experiments import (
    compare_retrieval_experiments,
)
from backend.evaluation.models import (
    EvaluationPopulation,
    EvaluationRunResult,
    QuestionEvaluationResult,
)

def make_question_result(
    question_id: str,
    rank: int | None,
) -> QuestionEvaluationResult:
    return QuestionEvaluationResult(
        question_id=question_id,
        question=f"Question {question_id}",
        first_relevant_rank=rank,
        retrieved_results=(),
    )

def make_run(
    ranks: tuple[int | None, ...],
    top_1: float,
    hit_at_k: float,
    mrr: float,
    top_k: int = 5,
    population: EvaluationPopulation | None = None,
) -> EvaluationRunResult:
    question_results = tuple(
        make_question_result(
            f"Q{index:03d}",
            rank,
        )
        for index, rank in enumerate(
            ranks,
            start=1,
        )
    )

    if population is None:
        population = EvaluationPopulation(
            dataset_questions=len(ranks),
            direct_answer_questions=len(ranks),
            grounded_overview_questions=0,
            clarify_questions=0,
            no_grounded_answer_questions=0,
        )

    return EvaluationRunResult(
        question_results=question_results,
        top_1_accuracy=top_1,
        hit_at_k=hit_at_k,
        mrr=mrr,
        top_k=top_k,
        pass_threshold=0.95,
        passed=top_1 >= 0.95,
        population=population,
    )

def make_config(
    dataset_status: DatasetValidationStatus = (
        DatasetValidationStatus.PRELIMINARY
    ),
) -> RetrievalExperimentConfig:
    return RetrievalExperimentConfig(
        experiment_name="Test comparison",
        baseline_name="Baseline",
        candidate_name="Candidate",
        dataset_name="Test dataset",
        dataset_status=dataset_status,
    )

def test_comparison_reports_metric_improvements():
    baseline = make_run(
        ranks=(1, 3, None),
        top_1=1.0 / 3.0,
        hit_at_k=2.0 / 3.0,
        mrr=(1.0 + (1.0 / 3.0)) / 3.0,
    )

    candidate = make_run(
        ranks=(1, 1, 1),
        top_1=1.0,
        hit_at_k=1.0,
        mrr=1.0,
    )

    comparison = compare_retrieval_experiments(
        baseline,
        candidate,
        make_config(),
    )

    assert comparison.top_1.delta > 0
    assert comparison.hit_at_k.delta > 0
    assert comparison.mrr.delta > 0
    assert (
        comparison.direction
        == ExperimentDirection.IMPROVED
    )

def test_comparison_retains_question_rank_changes():
    baseline = make_run(
        ranks=(1, 3, None),
        top_1=1.0 / 3.0,
        hit_at_k=2.0 / 3.0,
        mrr=0.4444,
    )

    candidate = make_run(
        ranks=(1, 1, 2),
        top_1=2.0 / 3.0,
        hit_at_k=1.0,
        mrr=0.8333,
    )

    comparison = compare_retrieval_experiments(
        baseline,
        candidate,
        make_config(),
    )

    assert comparison.question_rank_changes[0].baseline_rank == 1
    assert comparison.question_rank_changes[0].candidate_rank == 1

    assert comparison.question_rank_changes[1].baseline_rank == 3
    assert comparison.question_rank_changes[1].candidate_rank == 1

    assert comparison.question_rank_changes[2].baseline_rank is None
    assert comparison.question_rank_changes[2].candidate_rank == 2

def test_preliminary_dataset_cannot_satisfy_validated_gate():
    baseline = make_run(
        ranks=(1,),
        top_1=0.90,
        hit_at_k=1.0,
        mrr=0.90,
    )

    candidate = make_run(
        ranks=(1,),
        top_1=1.0,
        hit_at_k=1.0,
        mrr=1.0,
    )

    comparison = compare_retrieval_experiments(
        baseline,
        candidate,
        make_config(
            DatasetValidationStatus.PRELIMINARY
        ),
    )

    assert comparison.quality_gate_passed
    assert not comparison.validated_dataset_gate_passed
    assert (
    comparison.selection_decision
    == ExperimentSelectionDecision.REQUIRES_VALIDATED_EVALUATION
    )

def test_human_validated_dataset_can_satisfy_validated_gate():
    baseline = make_run(
        ranks=(1,),
        top_1=0.90,
        hit_at_k=1.0,
        mrr=0.90,
    )

    candidate = make_run(
        ranks=(1,),
        top_1=0.95,
        hit_at_k=1.0,
        mrr=0.95,
    )

    comparison = compare_retrieval_experiments(
        baseline,
        candidate,
        make_config(
            DatasetValidationStatus.HUMAN_VALIDATED
        ),
    )

    assert comparison.quality_gate_passed
    assert comparison.validated_dataset_gate_passed
    assert (
        comparison.direction
        == ExperimentDirection.IMPROVED
    )
    assert (
        comparison.selection_decision
        == ExperimentSelectionDecision.ELIGIBLE_FOR_SELECTION
    )

def test_candidate_below_95_percent_fails_quality_gate():
    baseline = make_run(
        ranks=(1,),
        top_1=0.80,
        hit_at_k=1.0,
        mrr=0.80,
    )

    candidate = make_run(
        ranks=(1,),
        top_1=0.949,
        hit_at_k=1.0,
        mrr=0.949,
    )

    comparison = compare_retrieval_experiments(
        baseline,
        candidate,
        make_config(
            DatasetValidationStatus.HUMAN_VALIDATED
        ),
    )

    assert not comparison.quality_gate_passed
    assert not comparison.validated_dataset_gate_passed
    assert (
    comparison.direction
    == ExperimentDirection.IMPROVED
    )
    assert (
        comparison.selection_decision
        == ExperimentSelectionDecision.REJECT_BELOW_THRESHOLD
    )

def test_top_1_regression_marks_experiment_as_regressed():
    baseline = make_run(
        ranks=(1, 1),
        top_1=1.0,
        hit_at_k=1.0,
        mrr=1.0,
    )

    candidate = make_run(
        ranks=(1, 2),
        top_1=0.5,
        hit_at_k=1.0,
        mrr=0.75,
    )

    comparison = compare_retrieval_experiments(
        baseline,
        candidate,
        make_config(),
    )

    assert (
        comparison.direction
        == ExperimentDirection.REGRESSED
    )

def test_identical_metrics_are_unchanged():
    baseline = make_run(
        ranks=(1,),
        top_1=1.0,
        hit_at_k=1.0,
        mrr=1.0,
    )

    candidate = make_run(
        ranks=(1,),
        top_1=1.0,
        hit_at_k=1.0,
        mrr=1.0,
    )

    comparison = compare_retrieval_experiments(
        baseline,
        candidate,
        make_config(),
    )

    assert (
        comparison.direction
        == ExperimentDirection.UNCHANGED
    )

def test_comparison_requires_same_top_k():
    baseline = make_run(
        ranks=(1,),
        top_1=1.0,
        hit_at_k=1.0,
        mrr=1.0,
        top_k=5,
    )

    candidate = make_run(
        ranks=(1,),
        top_1=1.0,
        hit_at_k=1.0,
        mrr=1.0,
        top_k=10,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Baseline and candidate must use the same top_k."
        ),
    ):
        compare_retrieval_experiments(
            baseline,
            candidate,
            make_config(),
        )

def test_comparison_requires_same_questions():
    baseline = make_run(
        ranks=(1, 1),
        top_1=1.0,
        hit_at_k=1.0,
        mrr=1.0,
    )

    candidate = make_run(
        ranks=(1,),
        top_1=1.0,
        hit_at_k=1.0,
        mrr=1.0,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Baseline and candidate must evaluate the same questions."
        ),
    ):
        compare_retrieval_experiments(
            baseline,
            candidate,
            make_config(),
        )

def test_validated_candidate_is_rejected_when_not_improved():
    baseline = make_run(
        ranks=(1,),
        top_1=1.0,
        hit_at_k=1.0,
        mrr=1.0,
    )

    candidate = make_run(
        ranks=(1,),
        top_1=0.95,
        hit_at_k=1.0,
        mrr=0.95,
    )

    comparison = compare_retrieval_experiments(
        baseline,
        candidate,
        make_config(
            DatasetValidationStatus.HUMAN_VALIDATED
        ),
    )

    assert comparison.quality_gate_passed
    assert comparison.validated_dataset_gate_passed

    assert (
        comparison.direction
        == ExperimentDirection.REGRESSED
    )

    assert (
        comparison.selection_decision
        == ExperimentSelectionDecision.REJECT_NOT_IMPROVED
    )

def test_comparison_preserves_evaluation_population():
    population = EvaluationPopulation(
        dataset_questions=30,
        direct_answer_questions=26,
        grounded_overview_questions=4,
        clarify_questions=0,
        no_grounded_answer_questions=0,
    )

    baseline = make_run(
        ranks=(1, 2),
        top_1=0.5,
        hit_at_k=1.0,
        mrr=0.75,
        population=population,
    )

    candidate = make_run(
        ranks=(1, 1),
        top_1=1.0,
        hit_at_k=1.0,
        mrr=1.0,
        population=population,
    )

    comparison = compare_retrieval_experiments(
        baseline,
        candidate,
        make_config(),
    )

    assert comparison.population == population
    assert comparison.population.dataset_questions == 30
    assert (
        comparison.population.direct_answer_questions
        == 26
    )
    assert (
        comparison.population.grounded_overview_questions
        == 4
    )

def test_comparison_rejects_different_evaluation_populations():
    baseline = make_run(
        ranks=(1, 1),
        top_1=1.0,
        hit_at_k=1.0,
        mrr=1.0,
        population=EvaluationPopulation(
            dataset_questions=30,
            direct_answer_questions=26,
            grounded_overview_questions=4,
            clarify_questions=0,
            no_grounded_answer_questions=0,
        ),
    )

    candidate = make_run(
        ranks=(1, 1),
        top_1=1.0,
        hit_at_k=1.0,
        mrr=1.0,
        population=EvaluationPopulation(
            dataset_questions=29,
            direct_answer_questions=26,
            grounded_overview_questions=3,
            clarify_questions=0,
            no_grounded_answer_questions=0,
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Baseline and candidate must use the "
            "same evaluation population."
        ),
    ):
        compare_retrieval_experiments(
            baseline,
            candidate,
            make_config(),
        )
