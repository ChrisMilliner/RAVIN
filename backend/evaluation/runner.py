from typing import Callable
from backend.evaluation.matching import (
    matches_expected_evidence,
)
from backend.evaluation.metrics import (
    calculate_hit_at_k,
    calculate_mrr,
    calculate_top_1_accuracy,
    meets_top_1_quality_gate,
)
from backend.evaluation.models import (
    EvaluationConfig,
    EvaluationQuestion,
    EvaluationRunResult,
    ExpectedEvidence,
    QuestionEvaluationResult,
)
from backend.retrieval.models import RetrievalResult

RetrievalFunction = Callable[
    [str, int],
    tuple[RetrievalResult, ...],
]

def find_first_relevant_rank(
    results: tuple[RetrievalResult, ...],
    expected_evidence: tuple[ExpectedEvidence, ...],
) -> int | None:
    for rank, result in enumerate(
        results,
        start=1,
    ):
        if any(
            matches_expected_evidence(
                result.chunk,
                expected,
            )
            for expected in expected_evidence
        ):
            return rank

    return None

def run_retrieval_evaluation(
    questions: tuple[EvaluationQuestion, ...],
    retrieve: RetrievalFunction,
    config: EvaluationConfig,
) -> EvaluationRunResult:
    if not questions:
        raise ValueError(
            "Cannot evaluate an empty question set."
        )

    question_results: list[
        QuestionEvaluationResult
    ] = []

    first_relevant_ranks: list[int | None] = []

    for question in questions:
        retrieved_results = retrieve(
            question.question,
            config.top_k,
        )

        retrieved_results = (
            retrieved_results[:config.top_k]
        )

        first_relevant_rank = (
            find_first_relevant_rank(
                retrieved_results,
                question.expected_evidence,
            )
        )

        first_relevant_ranks.append(
            first_relevant_rank
        )

        question_results.append(
            QuestionEvaluationResult(
                question_id=question.question_id,
                question=question.question,
                first_relevant_rank=(
                    first_relevant_rank
                ),
                retrieved_results=(
                    retrieved_results
                ),
            )
        )

    ranks = tuple(first_relevant_ranks)

    top_1_accuracy = calculate_top_1_accuracy(
        ranks
    )

    hit_at_k = calculate_hit_at_k(
        ranks,
        config.top_k,
    )

    mrr = calculate_mrr(
        ranks
    )

    passed = meets_top_1_quality_gate(
        top_1_accuracy,
        config.top_1_pass_threshold,
    )

    return EvaluationRunResult(
        question_results=tuple(question_results),
        top_1_accuracy=top_1_accuracy,
        hit_at_k=hit_at_k,
        mrr=mrr,
        top_k=config.top_k,
        pass_threshold=(
            config.top_1_pass_threshold
        ),
        passed=passed,
    )