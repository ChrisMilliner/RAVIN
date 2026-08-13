from dataclasses import dataclass
from backend.retrieval.models import RetrievalResult

DEFAULT_TOP_K = 5
DEFAULT_TOP_1_PASS_THRESHOLD = 0.95

@dataclass(frozen=True)
class ExpectedEvidence:
    policy_id: str
    heading_path: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError(
                "Expected evidence policy ID cannot be empty."
            )

        if not self.heading_path:
            raise ValueError(
                "Expected evidence heading path cannot be empty."
            )

@dataclass(frozen=True)
class EvaluationQuestion:
    question_id: str
    question: str
    expected_evidence: tuple[ExpectedEvidence, ...]
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError(
                "Evaluation question ID cannot be empty."
            )

        if not self.question.strip():
            raise ValueError(
                "Evaluation question cannot be empty."
            )

        if not self.expected_evidence:
            raise ValueError(
                "Evaluation question must define expected evidence."
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
class EvaluationRunResult:
    question_results: tuple[QuestionEvaluationResult, ...]
    top_1_accuracy: float
    hit_at_k: float
    mrr: float
    top_k: int
    pass_threshold: float
    passed: bool