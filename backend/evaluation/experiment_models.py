from dataclasses import dataclass
from enum import Enum

class DatasetValidationStatus(str, Enum):
    PRELIMINARY = "preliminary"
    HUMAN_VALIDATED = "human-validated"

class ExperimentDirection(str, Enum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    UNCHANGED = "unchanged"

class ExperimentSelectionDecision(str, Enum):
    REJECT_BELOW_THRESHOLD = (
        "reject-below-threshold"
    )
    REQUIRES_VALIDATED_EVALUATION = (
        "requires-validated-evaluation"
    )
    REJECT_NOT_IMPROVED = (
        "reject-not-improved"
    )
    ELIGIBLE_FOR_SELECTION = (
        "eligible-for-selection"
    )

@dataclass(frozen=True)
class RetrievalExperimentConfig:
    experiment_name: str
    baseline_name: str
    candidate_name: str
    dataset_name: str
    dataset_status: DatasetValidationStatus
    top_k: int = 5
    quality_threshold: float = 0.95

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError(
                "Experiment name cannot be empty."
            )

        if not self.baseline_name.strip():
            raise ValueError(
                "Baseline configuration name cannot be empty."
            )

        if not self.candidate_name.strip():
            raise ValueError(
                "Candidate configuration name cannot be empty."
            )

        if not self.dataset_name.strip():
            raise ValueError(
                "Evaluation dataset name cannot be empty."
            )

        if self.top_k <= 0:
            raise ValueError(
                "Experiment top_k must be greater than zero."
            )

        if not 0.0 <= self.quality_threshold <= 1.0:
            raise ValueError(
                "Experiment quality threshold must be between 0 and 1."
            )

@dataclass(frozen=True)
class MetricComparison:
    baseline: float
    candidate: float

    @property
    def delta(self) -> float:
        return self.candidate - self.baseline

@dataclass(frozen=True)
class QuestionRankChange:
    question_id: str
    baseline_rank: int | None
    candidate_rank: int | None

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError(
                "Question ID cannot be empty."
            )

        if (
            self.baseline_rank is not None
            and self.baseline_rank <= 0
        ):
            raise ValueError(
                "Baseline rank must be greater than zero."
            )

        if (
            self.candidate_rank is not None
            and self.candidate_rank <= 0
        ):
            raise ValueError(
                "Candidate rank must be greater than zero."
            )

@dataclass(frozen=True)
class RetrievalExperimentComparison:
    config: RetrievalExperimentConfig
    top_1: MetricComparison
    hit_at_k: MetricComparison
    mrr: MetricComparison
    question_rank_changes: tuple[QuestionRankChange, ...]
    direction: ExperimentDirection
    selection_decision: ExperimentSelectionDecision
    quality_gate_passed: bool
    validated_dataset_gate_passed: bool