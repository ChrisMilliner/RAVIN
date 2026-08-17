import pytest
from backend.evaluation.models import (
    EvaluationBehavior,
    EvaluationConfig,
    EvaluationQuestion,
    ExpectedEvidence,
)
from backend.evaluation.runner import (
    find_first_relevant_rank,
    run_retrieval_evaluation,
)
from backend.ingestion.models import PolicyChunk
from backend.retrieval.models import RetrievalResult

def make_chunk(
    policy_id: str,
    heading_path: tuple[str, ...],
) -> PolicyChunk:
    return PolicyChunk(
        policy_id=policy_id,
        policy_title="Test Policy",
        source_url=(
            "https://policies.latrobe.edu.au/"
            f"document/view.php?id={policy_id}"
        ),
        status="Current",
        effective_date=None,
        review_date=None,
        chunk_index=0,
        text="Test evidence.",
        heading_path=heading_path,
    )

def make_result(
    policy_id: str,
    heading_path: tuple[str, ...],
    score: float,
) -> RetrievalResult:
    return RetrievalResult(
        chunk=make_chunk(
            policy_id,
            heading_path,
        ),
        score=score,
    )

def make_question(
    question_id: str,
    question: str,
    behavior: EvaluationBehavior = (
        EvaluationBehavior.DIRECT_ANSWER
    ),
) -> EvaluationQuestion:
    return EvaluationQuestion(
        question_id=question_id,
        question=question,
        expected_evidence=(
            ExpectedEvidence(
                policy_id="208",
                heading_path=(
                    "Section 4 - Key Decisions",
                ),
            ),
        ),
        behavior=behavior,
    )

def test_find_first_relevant_rank_returns_first_match():
    results = (
        make_result(
            "169",
            ("Section 5 - Policy Statement",),
            0.9,
        ),
        make_result(
            "208",
            ("Section 4 - Key Decisions",),
            0.8,
        ),
        make_result(
            "208",
            ("Section 4 - Key Decisions",),
            0.7,
        ),
    )

    expected = (
        ExpectedEvidence(
            policy_id="208",
            heading_path=(
                "Section 4 - Key Decisions",
            ),
        ),
    )

    assert find_first_relevant_rank(
        results,
        expected,
    ) == 2

def test_find_first_relevant_rank_returns_none_when_missing():
    results = (
        make_result(
            "169",
            ("Section 5 - Policy Statement",),
            0.9,
        ),
    )

    expected = (
        ExpectedEvidence(
            policy_id="208",
            heading_path=(
                "Section 4 - Key Decisions",
            ),
        ),
    )

    assert find_first_relevant_rank(
        results,
        expected,
    ) is None

def test_runner_calculates_metrics_from_ranked_results():
    questions = (
        make_question("Q001", "Question one"),
        make_question("Q002", "Question two"),
        make_question("Q003", "Question three"),
    )

    results_by_question = {
        "Question one": (
            make_result(
                "208",
                ("Section 4 - Key Decisions",),
                0.9,
            ),
        ),
        "Question two": (
            make_result(
                "169",
                ("Section 5 - Policy Statement",),
                0.9,
            ),
            make_result(
                "169",
                ("Section 6 - Procedures",),
                0.8,
            ),
            make_result(
                "208",
                ("Section 4 - Key Decisions",),
                0.7,
            ),
        ),
        "Question three": (
            make_result(
                "169",
                ("Section 5 - Policy Statement",),
                0.9,
            ),
        ),
    }

    def retrieve(
        question: str,
        top_k: int,
    ) -> tuple[RetrievalResult, ...]:
        return results_by_question[question][:top_k]

    result = run_retrieval_evaluation(
        questions,
        retrieve,
        EvaluationConfig(top_k=5),
    )

    assert result.top_1_accuracy == pytest.approx(
        1.0 / 3.0
    )

    assert result.hit_at_k == pytest.approx(
        2.0 / 3.0
    )

    assert result.mrr == pytest.approx(
        (1.0 + (1.0 / 3.0)) / 3.0
    )

    assert not result.passed

def test_runner_reports_pass_for_perfect_retrieval():
    questions = (
        make_question("Q001", "Question one"),
        make_question("Q002", "Question two"),
    )

    def retrieve(
        question: str,
        top_k: int,
    ) -> tuple[RetrievalResult, ...]:
        return (
            make_result(
                "208",
                ("Section 4 - Key Decisions",),
                0.9,
            ),
        )

    result = run_retrieval_evaluation(
        questions,
        retrieve,
        EvaluationConfig(),
    )

    assert result.top_1_accuracy == 1.0
    assert result.passed

def test_runner_retains_per_question_failure_details():
    question = make_question(
        "Q001",
        "Question one",
    )

    def retrieve(
        question_text: str,
        top_k: int,
    ) -> tuple[RetrievalResult, ...]:
        return (
            make_result(
                "169",
                ("Section 7 - Definitions",),
                0.7,
            ),
        )

    result = run_retrieval_evaluation(
        (question,),
        retrieve,
        EvaluationConfig(),
    )

    question_result = result.question_results[0]

    assert question_result.question_id == "Q001"
    assert question_result.first_relevant_rank is None
    assert len(question_result.retrieved_results) == 1

def test_runner_rejects_empty_question_set():
    def retrieve(
        question: str,
        top_k: int,
    ) -> tuple[RetrievalResult, ...]:
        return ()

    with pytest.raises(
        ValueError,
        match="Cannot evaluate an empty question set.",
    ):
        run_retrieval_evaluation(
            (),
            retrieve,
            EvaluationConfig(),
        )

def test_runner_evaluates_only_direct_answer_questions():
    questions = (
        make_question(
            "Q001",
            "Direct question",
        ),
        make_question(
            "Q002",
            "Overview question",
            EvaluationBehavior.GROUNDED_OVERVIEW,
        ),
        make_question(
            "Q003",
            "Clarification question",
            EvaluationBehavior.CLARIFY,
        ),
    )

    retrieved_questions: list[str] = []

    def retrieve(
        question: str,
        top_k: int,
    ) -> tuple[RetrievalResult, ...]:
        retrieved_questions.append(question)

        return (
            make_result(
                "208",
                ("Section 4 - Key Decisions",),
                0.9,
            ),
        )

    result = run_retrieval_evaluation(
        questions,
        retrieve,
        EvaluationConfig(),
    )

    assert retrieved_questions == [
        "Direct question",
    ]

    assert len(result.question_results) == 1

    assert (
        result.question_results[0].question_id
        == "Q001"
    )

    assert result.top_1_accuracy == 1.0
    assert result.hit_at_k == 1.0
    assert result.mrr == 1.0
    assert result.passed

def test_runner_rejects_question_set_without_direct_answers():
    questions = (
        make_question(
            "Q001",
            "Overview question",
            EvaluationBehavior.GROUNDED_OVERVIEW,
        ),
        make_question(
            "Q002",
            "Clarification question",
            EvaluationBehavior.CLARIFY,
        ),
    )

    def retrieve(
        question: str,
        top_k: int,
    ) -> tuple[RetrievalResult, ...]:
        return ()

    with pytest.raises(
        ValueError,
        match=(
            "Retrieval ranking evaluation requires "
            "at least one Direct Answer question."
        ),
    ):
        run_retrieval_evaluation(
            questions,
            retrieve,
            EvaluationConfig(),
        )
