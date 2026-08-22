from dataclasses import dataclass
from enum import Enum
from backend.behavior import AnswerBehavior

class QuestionIntent(str, Enum):
    FOCUSED = "focused"
    BROAD = "broad"
    AMBIGUOUS = "ambiguous"


class EvidenceSufficiency(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    UNCERTAIN = "uncertain"

@dataclass(frozen=True)
class QuestionAssessment:
    intent: QuestionIntent
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.intent,
            QuestionIntent,
        ):
            raise ValueError(
                "Question intent must be a "
                "QuestionIntent."
            )

        if not self.reason.strip():
            raise ValueError(
                "Question assessment reason "
                "cannot be empty."
            )

@dataclass(frozen=True)
class EvidenceSignals:
    retrieved_count: int
    context_block_count: int
    distinct_policy_count: int
    top_score: float | None
    second_score: float | None
    score_margin: float | None

    def __post_init__(self) -> None:
        if self.retrieved_count < 0:
            raise ValueError(
                "Retrieved evidence count cannot "
                "be negative."
            )

        if self.context_block_count < 0:
            raise ValueError(
                "Context block count cannot "
                "be negative."
            )

        if self.distinct_policy_count < 0:
            raise ValueError(
                "Distinct policy count cannot "
                "be negative."
            )

        if (
            self.distinct_policy_count
            > self.context_block_count
        ):
            raise ValueError(
                "Distinct policy count cannot exceed "
                "the context block count."
            )

        if self.retrieved_count == 0:
            if (
                self.top_score is not None
                or self.second_score is not None
                or self.score_margin is not None
            ):
                raise ValueError(
                    "Empty retrieval cannot define "
                    "retrieval scores."
                )

        elif self.retrieved_count == 1:
            if self.top_score is None:
                raise ValueError(
                    "Single-result retrieval must "
                    "define a top score."
                )

            if (
                self.second_score is not None
                or self.score_margin is not None
            ):
                raise ValueError(
                    "Single-result retrieval cannot "
                    "define a second score or margin."
                )

        else:
            if (
                self.top_score is None
                or self.second_score is None
                or self.score_margin is None
            ):
                raise ValueError(
                    "Multi-result retrieval must "
                    "define top, second and margin "
                    "scores."
                )

    @property
    def has_evidence(self) -> bool:
        return (
            self.retrieved_count > 0
            and self.context_block_count > 0
        )

@dataclass(frozen=True)
class EvidenceAssessment:
    sufficiency: EvidenceSufficiency
    signals: EvidenceSignals
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.sufficiency,
            EvidenceSufficiency,
        ):
            raise ValueError(
                "Evidence sufficiency must be an "
                "EvidenceSufficiency."
            )

        if not isinstance(
            self.signals,
            EvidenceSignals,
        ):
            raise ValueError(
                "Evidence assessment signals must "
                "be EvidenceSignals."
            )

        if not self.reason.strip():
            raise ValueError(
                "Evidence assessment reason "
                "cannot be empty."
            )

@dataclass(frozen=True)
class RoutingResult:
    behavior: AnswerBehavior
    question_assessment: QuestionAssessment
    evidence_assessment: EvidenceAssessment
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.behavior,
            AnswerBehavior,
        ):
            raise ValueError(
                "Routing behavior must be an "
                "AnswerBehavior."
            )

        if not isinstance(
            self.question_assessment,
            QuestionAssessment,
        ):
            raise ValueError(
                "Routing question assessment must "
                "be QuestionAssessment."
            )

        if not isinstance(
            self.evidence_assessment,
            EvidenceAssessment,
        ):
            raise ValueError(
                "Routing evidence assessment must "
                "be EvidenceAssessment."
            )

        if not self.reason.strip():
            raise ValueError(
                "Routing reason cannot be empty."
            )