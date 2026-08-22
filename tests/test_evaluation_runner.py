import pytest
from backend.evaluation.models import (
    EvaluationBehavior,
    EvaluationConfig,
    EvaluationQuestion,
    ExpectedEvidence,
    ExpectedEvidenceGroup,
)
from backend.evaluation.runner import (
    evaluate_grounded_overview_question,
    find_first_relevant_rank,
    run_retrieval_evaluation,
)
from backend.ingestion.models import PolicyChunk
from backend.retrieval.models import RetrievalResult

def make_chunk(
    policy_id: str,
    heading_path: tuple[str, ...],
    text: str = "Test evidence.",
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
        text=text,
        heading_path=heading_path,
    )

def make_result(
    policy_id: str,
    heading_path: tuple[str, ...],
    score: float,
    text: str = "Test evidence.",
) -> RetrievalResult:
    return RetrievalResult(
        chunk=make_chunk(
            policy_id,
            heading_path,
            text,
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

def make_overview_question() -> EvaluationQuestion:
    return EvaluationQuestion(
        question_id="RB002",
        question=(
            "What happens when a student is not "
            "making satisfactory academic progress?"
        ),
        expected_evidence=(
            ExpectedEvidence(
                policy_id="220",
                heading_path=(
                    "Section 6 - Procedures",
                ),
            ),
        ),
        behavior=(
            EvaluationBehavior.GROUNDED_OVERVIEW
        ),
        expected_evidence_groups=(
            ExpectedEvidenceGroup(
                group_id="progression_stages",
                description=(
                    "Subject failure triggers "
                    "staged progression."
                ),
                alternatives=(
                    ExpectedEvidence(
                        policy_id="220",
                        heading_path=(
                            "Section 6 - Procedures",
                            (
                                "Part A - Monitoring and "
                                "Determining Academic Progression"
                            ),
                        ),
                        text_contains=(
                            "three stages of "
                            "academic progression"
                        ),
                    ),
                ),
            ),
            ExpectedEvidenceGroup(
                group_id="support_and_interventions",
                description=(
                    "Progression includes support "
                    "and interventions."
                ),
                alternatives=(
                    ExpectedEvidence(
                        policy_id="220",
                        heading_path=(
                            "Section 5 - Policy Statement",
                        ),
                        text_contains=(
                            "academic and "
                            "non-academic support"
                        ),
                    ),
                    ExpectedEvidence(
                        policy_id="220",
                        heading_path=(
                            "Section 6 - Procedures",
                            (
                                "Part A - Monitoring and "
                                "Determining Academic Progression"
                            ),
                        ),
                        text_contains=(
                            "associated support "
                            "and interventions"
                        ),
                    ),
                ),
            ),
            ExpectedEvidenceGroup(
                group_id=(
                    "escalation_and_consequences"
                ),
                description=(
                    "Progression may escalate to "
                    "show cause and exclusion."
                ),
                alternatives=(
                    ExpectedEvidence(
                        policy_id="220",
                        heading_path=(
                            "Section 6 - Procedures",
                            (
                                "Part A - Monitoring and "
                                "Determining Academic Progression"
                            ),
                        ),
                        text_contains=(
                            "Show Cause and "
                            "Course Exclusion"
                        ),
                    ),
                ),
            ),
        ),
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

def test_runner_reports_behavior_population():
    no_answer_question = EvaluationQuestion(
        question_id="Q004",
        question="Unsupported question",
        expected_evidence=(),
        behavior=(
            EvaluationBehavior.NO_GROUNDED_ANSWER
        ),
    )

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
        no_answer_question,
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

    assert result.population.dataset_questions == 4
    assert (
        result.population.direct_answer_questions
        == 1
    )
    assert (
        result.population.grounded_overview_questions
        == 1
    )
    assert result.population.clarify_questions == 1
    assert (
        result.population.no_grounded_answer_questions
        == 1
    )

    assert len(result.question_results) == 1

def test_grounded_overview_question_covers_all_required_groups():
    question = make_overview_question()

    results = (
        make_result(
            "220",
            (
                "Section 6 - Procedures",
                (
                    "Part A - Monitoring and "
                    "Determining Academic Progression"
                ),
            ),
            0.95,
            (
                "Students trigger one of three stages "
                "of academic progression with associated "
                "support and interventions."
            ),
        ),
        make_result(
            "220",
            (
                "Section 6 - Procedures",
                (
                    "Part A - Monitoring and "
                    "Determining Academic Progression"
                ),
            ),
            0.90,
            (
                "Academic Progression Stage Three "
                "involves Show Cause and Course Exclusion."
            ),
        ),
    )

    result = evaluate_grounded_overview_question(
        question,
        results,
    )

    assert result.question_id == "RB002"
    assert result.total_groups == 3
    assert result.covered_groups == 3
    assert result.evidence_coverage == 1.0
    assert result.passed is True

    assert tuple(
        group.group_id
        for group in result.group_results
    ) == (
        "progression_stages",
        "support_and_interventions",
        "escalation_and_consequences",
    )

    assert all(
        group.covered
        for group in result.group_results
    )

def test_grounded_overview_question_fails_when_required_group_missing():
    question = make_overview_question()

    results = (
        make_result(
            "220",
            (
                "Section 6 - Procedures",
                (
                    "Part A - Monitoring and "
                    "Determining Academic Progression"
                ),
            ),
            0.95,
            (
                "Students trigger one of three stages "
                "of academic progression with associated "
                "support and interventions."
            ),
        ),
    )

    result = evaluate_grounded_overview_question(
        question,
        results,
    )

    assert result.total_groups == 3
    assert result.covered_groups == 2
    assert result.evidence_coverage == pytest.approx(
        2.0 / 3.0
    )
    assert result.passed is False

    coverage = {
        group.group_id: group.covered
        for group in result.group_results
    }

    assert coverage == {
        "progression_stages": True,
        "support_and_interventions": True,
        "escalation_and_consequences": False,
    }

def test_grounded_overview_question_accepts_alternative_evidence():
    question = make_overview_question()

    results = (
        make_result(
            "220",
            (
                "Section 5 - Policy Statement",
            ),
            0.90,
            (
                "Students have multiple avenues for "
                "academic and non-academic support."
            ),
        ),
    )

    result = evaluate_grounded_overview_question(
        question,
        results,
    )

    coverage = {
        group.group_id: group.covered
        for group in result.group_results
    }

    assert (
        coverage["support_and_interventions"]
        is True
    )

def test_grounded_overview_question_handles_empty_retrieval():
    question = make_overview_question()

    result = evaluate_grounded_overview_question(
        question,
        (),
    )

    assert result.total_groups == 3
    assert result.covered_groups == 0
    assert result.evidence_coverage == 0.0
    assert result.passed is False

    assert all(
        not group.covered
        for group in result.group_results
    )

def test_grounded_overview_question_rejects_wrong_behavior():
    question = make_question(
        "Q001",
        "Direct question",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Grounded overview evaluation requires "
            "a Grounded Overview question."
        ),
    ):
        evaluate_grounded_overview_question(
            question,
            (),
        )

def test_grounded_overview_question_requires_evidence_groups():
    question = EvaluationQuestion(
        question_id="Q002",
        question="Overview question",
        expected_evidence=(
            ExpectedEvidence(
                policy_id="220",
                heading_path=(
                    "Section 5 - Policy Statement",
                ),
            ),
        ),
        behavior=(
            EvaluationBehavior.GROUNDED_OVERVIEW
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Grounded Overview question must define "
            "expected evidence groups."
        ),
    ):
        evaluate_grounded_overview_question(
            question,
            (),
        )
