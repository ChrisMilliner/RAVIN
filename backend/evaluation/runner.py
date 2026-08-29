"""
Execute retrieval and grounded-overview evaluation runs.

This module evaluates ranked retrieval results against structured
expected evidence, calculates question-level ranks and aggregate
metrics, and assesses evidence-group coverage for broad grounded
overview questions.

The runner reports measured performance without changing retrieval
configuration or promoting development results to validated status.
"""

from typing import Callable
from backend.evaluation.matching import (
    is_expected_evidence_group_covered,
    matches_expected_evidence,
)
from backend.evaluation.metrics import (
    calculate_hit_at_k,
    calculate_mrr,
    calculate_top_1_accuracy,
    meets_top_1_quality_gate,
)
from backend.evaluation.models import (
    EvaluationBehavior,
    EvaluationConfig,
    EvaluationPopulation,
    EvaluationQuestion,
    EvaluationRunResult,
    ExpectedEvidence,
    GroundedOverviewEvaluationResult,
    GroundedOverviewGroupResult,
    GroundedOverviewQuestionResult,
    QuestionEvaluationResult,
    GroundedOverviewEvaluationConfig,
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

def evaluate_grounded_overview_question(
    question: EvaluationQuestion,
    retrieved_results: tuple[
        RetrievalResult,
        ...
    ],
) -> GroundedOverviewQuestionResult:
    if (
        question.behavior
        != EvaluationBehavior.GROUNDED_OVERVIEW
    ):
        raise ValueError(
            "Grounded overview evaluation requires "
            "a Grounded Overview question."
        )

    if not question.expected_evidence_groups:
        raise ValueError(
            "Grounded Overview question must define "
            "expected evidence groups."
        )

    chunks = tuple(
        result.chunk
        for result in retrieved_results
    )

    group_results = tuple(
        GroundedOverviewGroupResult(
            group_id=group.group_id,
            covered=(
                is_expected_evidence_group_covered(
                    chunks,
                    group,
                )
            ),
        )
        for group in question.expected_evidence_groups
    )

    return GroundedOverviewQuestionResult(
        question_id=question.question_id,
        group_results=group_results,
    )

def run_grounded_overview_evaluation(
    questions: tuple[EvaluationQuestion, ...],
    retrieve: RetrievalFunction,
    config: GroundedOverviewEvaluationConfig,
) -> GroundedOverviewEvaluationResult:
    overview_questions = tuple(
        question
        for question in questions
        if (
            question.behavior
            == EvaluationBehavior.GROUNDED_OVERVIEW
        )
    )

    if not overview_questions:
        raise ValueError(
            "Grounded overview evaluation requires "
            "at least one Grounded Overview question."
        )

    question_results: list[
        GroundedOverviewQuestionResult
    ] = []

    for question in overview_questions:
        retrieved_results = retrieve(
            question.question,
            config.top_k,
        )

        retrieved_results = (
            retrieved_results[:config.top_k]
        )

        question_results.append(
            evaluate_grounded_overview_question(
                question,
                retrieved_results,
            )
        )

    return GroundedOverviewEvaluationResult(
        question_results=tuple(
            question_results
        ),
        pass_threshold=config.pass_threshold,
    )

def run_retrieval_evaluation(
    questions: tuple[EvaluationQuestion, ...],
    retrieve: RetrievalFunction,
    config: EvaluationConfig,
) -> EvaluationRunResult:
    if not questions:
        raise ValueError(
            "Cannot evaluate an empty question set."
        )

    population = EvaluationPopulation(
        dataset_questions=len(questions),
        direct_answer_questions=sum(
            1
            for question in questions
            if (
                question.behavior
                == EvaluationBehavior.DIRECT_ANSWER
            )
        ),
        grounded_overview_questions=sum(
            1
            for question in questions
            if (
                question.behavior
                == EvaluationBehavior.GROUNDED_OVERVIEW
            )
        ),
        clarify_questions=sum(
            1
            for question in questions
            if (
                question.behavior
                == EvaluationBehavior.CLARIFY
            )
        ),
        no_grounded_answer_questions=sum(
            1
            for question in questions
            if (
                question.behavior
                == EvaluationBehavior.NO_GROUNDED_ANSWER
            )
        ),
    )

    ranking_questions = tuple(
        question
        for question in questions
        if (
            question.behavior
            == EvaluationBehavior.DIRECT_ANSWER
        )
    )

    if not ranking_questions:
        raise ValueError(
            "Retrieval ranking evaluation requires "
            "at least one Direct Answer question."
        )

    question_results: list[
        QuestionEvaluationResult
    ] = []

    first_relevant_ranks: list[int | None] = []

    for question in ranking_questions:
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
        population=population,
    )