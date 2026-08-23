from dataclasses import dataclass
from enum import Enum
from backend.retrieval.models import RetrievalResult

DEFAULT_TOP_K = 5
DEFAULT_TOP_1_PASS_THRESHOLD = 0.95
DEFAULT_GROUNDED_OVERVIEW_PASS_THRESHOLD = 0.95

class EvaluationBehavior(str, Enum):
    DIRECT_ANSWER = "direct_answer"
    GROUNDED_OVERVIEW = "grounded_overview"
    CLARIFY = "clarify"
    NO_GROUNDED_ANSWER = "no_grounded_answer"

@dataclass(frozen=True)
class ExpectedEvidence:
    policy_id: str
    heading_path: tuple[str, ...]
    allow_descendants: bool = False
    text_contains: str | None = None

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError(
                "Expected evidence policy ID cannot be empty."
            )

        if not self.heading_path:
            raise ValueError(
                "Expected evidence heading path cannot be empty."
            )

        if (
            self.text_contains is not None
            and not self.text_contains.strip()
        ):
            raise ValueError(
                "Expected evidence text fragment cannot be empty."
            )

@dataclass(frozen=True)
class ExpectedEvidenceGroup:
    group_id: str
    description: str
    alternatives: tuple[ExpectedEvidence, ...]

    def __post_init__(self) -> None:
        if not self.group_id.strip():
            raise ValueError(
                "Expected evidence group ID "
                "cannot be empty."
            )

        if not self.description.strip():
            raise ValueError(
                "Expected evidence group description "
                "cannot be empty."
            )

        if not self.alternatives:
            raise ValueError(
                "Expected evidence group must define "
                "at least one evidence alternative."
            )

        if any(
            not isinstance(
                alternative,
                ExpectedEvidence,
            )
            for alternative in self.alternatives
        ):
            raise ValueError(
                "Expected evidence group alternatives "
                "must be ExpectedEvidence values."
            )

@dataclass(frozen=True)
class GroundedOverviewGroupResult:
    group_id: str
    covered: bool

    def __post_init__(self) -> None:
        if not self.group_id.strip():
            raise ValueError(
                "Grounded overview group result ID "
                "cannot be empty."
            )

        if not isinstance(
            self.covered,
            bool,
        ):
            raise ValueError(
                "Grounded overview group result "
                "covered must be a boolean."
            )

@dataclass(frozen=True)
class GroundedOverviewQuestionResult:
    question_id: str
    group_results: tuple[
        GroundedOverviewGroupResult,
        ...
    ]

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError(
                "Grounded overview question result ID "
                "cannot be empty."
            )

        if not self.group_results:
            raise ValueError(
                "Grounded overview question result "
                "must contain at least one group result."
            )

        if any(
            not isinstance(
                result,
                GroundedOverviewGroupResult,
            )
            for result in self.group_results
        ):
            raise ValueError(
                "Grounded overview question group "
                "results must be "
                "GroundedOverviewGroupResult values."
            )

        group_ids = [
            result.group_id
            for result in self.group_results
        ]

        if len(group_ids) != len(set(group_ids)):
            raise ValueError(
                "Grounded overview question result "
                "cannot contain duplicate group IDs."
            )

    @property
    def total_groups(self) -> int:
        return len(self.group_results)

    @property
    def covered_groups(self) -> int:
        return sum(
            1
            for result in self.group_results
            if result.covered
        )

    @property
    def evidence_coverage(self) -> float:
        return (
            self.covered_groups
            / self.total_groups
        )

    @property
    def passed(self) -> bool:
        return (
            self.covered_groups
            == self.total_groups
        )

@dataclass(frozen=True)
class GroundedOverviewEvaluationConfig:
    top_k: int = DEFAULT_TOP_K
    pass_threshold: float = (
        DEFAULT_GROUNDED_OVERVIEW_PASS_THRESHOLD
    )

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError(
                "Grounded overview top_k must be "
                "greater than zero."
            )

        if not 0.0 <= self.pass_threshold <= 1.0:
            raise ValueError(
                "Grounded overview pass threshold "
                "must be between 0 and 1."
            )

@dataclass(frozen=True)
class GroundedOverviewEvaluationResult:
    question_results: tuple[
        GroundedOverviewQuestionResult,
        ...
    ]
    pass_threshold: float

    def __post_init__(self) -> None:
        if not self.question_results:
            raise ValueError(
                "Grounded overview evaluation "
                "requires at least one question result."
            )

        if any(
            not isinstance(
                result,
                GroundedOverviewQuestionResult,
            )
            for result in self.question_results
        ):
            raise ValueError(
                "Grounded overview evaluation question "
                "results must be "
                "GroundedOverviewQuestionResult values."
            )

        question_ids = [
            result.question_id
            for result in self.question_results
        ]

        if len(question_ids) != len(
            set(question_ids)
        ):
            raise ValueError(
                "Grounded overview evaluation cannot "
                "contain duplicate question IDs."
            )

        if not 0.0 <= self.pass_threshold <= 1.0:
            raise ValueError(
                "Grounded overview pass threshold "
                "must be between 0 and 1."
            )

    @property
    def total_questions(self) -> int:
        return len(self.question_results)

    @property
    def passed_questions(self) -> int:
        return sum(
            1
            for result in self.question_results
            if result.passed
        )

    @property
    def question_pass_rate(self) -> float:
        return (
            self.passed_questions
            / self.total_questions
        )

    @property
    def total_groups(self) -> int:
        return sum(
            result.total_groups
            for result in self.question_results
        )

    @property
    def covered_groups(self) -> int:
        return sum(
            result.covered_groups
            for result in self.question_results
        )

    @property
    def evidence_group_coverage(self) -> float:
        return (
            self.covered_groups
            / self.total_groups
        )

    @property
    def passed(self) -> bool:
        return (
            self.question_pass_rate
            >= self.pass_threshold
        )

@dataclass(frozen=True)
class EvaluationQuestion:
    question_id: str
    question: str
    expected_evidence: tuple[ExpectedEvidence, ...]
    notes: str | None = None
    behavior: EvaluationBehavior = (
        EvaluationBehavior.DIRECT_ANSWER
    )
    expected_evidence_groups: tuple[
        ExpectedEvidenceGroup,
        ...
    ] = ()

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError(
                "Evaluation question ID cannot be empty."
            )

        if not self.question.strip():
            raise ValueError(
                "Evaluation question cannot be empty."
            )

        if not isinstance(
            self.behavior,
            EvaluationBehavior,
        ):
            raise ValueError(
                "Evaluation question behavior must be "
                "an EvaluationBehavior."
            )

        if (
            self.behavior
            == EvaluationBehavior.NO_GROUNDED_ANSWER
        ):
            if self.expected_evidence:
                raise ValueError(
                    "No-grounded-answer questions "
                    "must not define expected evidence."
                )
        elif not self.expected_evidence:
            raise ValueError(
                "Evaluation question must define "
                "expected evidence."
            )

        if any(
            not isinstance(
                group,
                ExpectedEvidenceGroup,
            )
            for group in self.expected_evidence_groups
        ):
            raise ValueError(
                "Evaluation question evidence groups "
                "must be ExpectedEvidenceGroup values."
            )

@dataclass(frozen=True)
class EvaluationConfig:
    top_k: int = DEFAULT_TOP_K
    top_1_pass_threshold: float = (
        DEFAULT_TOP_1_PASS_THRESHOLD
    )

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError(
                "Evaluation top_k must be greater than zero."
            )

        if not 0.0 <= self.top_1_pass_threshold <= 1.0:
            raise ValueError(
                "Top-1 pass threshold must be between 0 and 1."
            )

@dataclass(frozen=True)
class QuestionEvaluationResult:
    question_id: str
    question: str
    first_relevant_rank: int | None
    retrieved_results: tuple[RetrievalResult, ...]

@dataclass(frozen=True)
class EvaluationPopulation:
    dataset_questions: int
    direct_answer_questions: int
    grounded_overview_questions: int
    clarify_questions: int
    no_grounded_answer_questions: int

    def __post_init__(self) -> None:
        counts = (
            self.direct_answer_questions,
            self.grounded_overview_questions,
            self.clarify_questions,
            self.no_grounded_answer_questions,
        )

        if self.dataset_questions <= 0:
            raise ValueError(
                "Evaluation population must contain "
                "at least one dataset question."
            )

        if any(
            count < 0
            for count in counts
        ):
            raise ValueError(
                "Evaluation population counts "
                "cannot be negative."
            )

        if sum(counts) != self.dataset_questions:
            raise ValueError(
                "Evaluation population behavior counts "
                "must equal the dataset question count."
            )

@dataclass(frozen=True)
class EvaluationRunResult:
    question_results: tuple[QuestionEvaluationResult, ...]
    top_1_accuracy: float
    hit_at_k: float
    mrr: float
    top_k: int
    pass_threshold: float
    passed: bool
    population: EvaluationPopulation
