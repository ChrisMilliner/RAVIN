"""
Define proposition-level evidence coverage results.

Coverage records how strongly each material proposition is supported by
retrieved policy context and provides aggregate information used by the
deterministic evidence-sufficiency assessor.

Coverage thresholds are configurable development or validated control
parameters and must not be represented as system accuracy results.
"""

from dataclasses import dataclass
from enum import Enum
from backend.routing.material_propositions import (
    MaterialProposition,
)

class PropositionCoverageStatus(
    str,
    Enum,
):
    COVERED = "covered"
    PARTIAL = "partial"
    UNCOVERED = "uncovered"

@dataclass(frozen=True)
class PropositionEvidenceCoverage:
    proposition: MaterialProposition
    status: PropositionCoverageStatus

    score: float

    evidence_indexes: tuple[
        int,
        ...
    ] = ()

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.proposition,
            MaterialProposition,
        ):
            raise ValueError(
                "Coverage proposition must be "
                "a MaterialProposition."
            )

        if not isinstance(
            self.status,
            PropositionCoverageStatus,
        ):
            raise ValueError(
                "Coverage status must be a "
                "PropositionCoverageStatus."
            )

        if not isinstance(
            self.score,
            float,
        ):
            raise ValueError(
                "Coverage score must be a float."
            )

        if not (
            0.0
            <= self.score
            <= 1.0
        ):
            raise ValueError(
                "Coverage score must be between "
                "0.0 and 1.0."
            )

        if not isinstance(
            self.evidence_indexes,
            tuple,
        ):
            raise ValueError(
                "Evidence indexes must be a tuple."
            )

        if not all(
            isinstance(
                index,
                int,
            )
            and index >= 0
            for index
            in self.evidence_indexes
        ):
            raise ValueError(
                "Evidence indexes must contain "
                "non-negative integers."
            )

        if (
            self.status
            == PropositionCoverageStatus.COVERED
            and not self.evidence_indexes
        ):
            raise ValueError(
                "Covered propositions must "
                "identify supporting evidence."
            )

@dataclass(frozen=True)
class QuestionEvidenceCoverage:
    propositions: tuple[
        PropositionEvidenceCoverage,
        ...
    ]

    def __post_init__(
        self,
    ) -> None:
        if not self.propositions:
            raise ValueError(
                "Question evidence coverage "
                "cannot be empty."
            )

        if not all(
            isinstance(
                coverage,
                PropositionEvidenceCoverage,
            )
            for coverage
            in self.propositions
        ):
            raise ValueError(
                "Question evidence coverage must "
                "contain proposition coverage values."
            )

    @property
    def all_covered(
        self,
    ) -> bool:
        return all(
            coverage.status
            == PropositionCoverageStatus.COVERED
            for coverage
            in self.propositions
        )

    @property
    def minimum_score(
        self,
    ) -> float:
        return min(
            coverage.score
            for coverage
            in self.propositions
        )