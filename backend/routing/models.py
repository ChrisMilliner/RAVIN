from dataclasses import dataclass
from backend.behavior import AnswerBehavior

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
class RoutingResult:
    behavior: AnswerBehavior
    evidence: EvidenceSignals
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
            self.evidence,
            EvidenceSignals,
        ):
            raise ValueError(
                "Routing evidence must be "
                "EvidenceSignals."
            )

        if not self.reason.strip():
            raise ValueError(
                "Routing reason cannot be empty."
            )