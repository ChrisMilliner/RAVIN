"""
Assess retrieved evidence against each material question proposition.

This module scores propositions against grounded retrieval context using
the configured AnswerabilityProvider and classifies each proposition as
covered, partially covered, or uncovered according to configured
thresholds.

The resulting coverage is passed to deterministic evidence-sufficiency
logic rather than directly selecting an answer behaviour.
"""

from backend.routing.answerability import (
    AnswerabilityProvider,
    score_answerability,
)
from backend.routing.material_propositions import (
    MaterialProposition,
    MaterialQuestionPropositions,
)
from backend.routing.proposition_coverage import (
    PropositionCoverageStatus,
    PropositionEvidenceCoverage,
    QuestionEvidenceCoverage,
)

class PropositionCoverageAssessor:
    def __init__(
        self,
        answerability_provider: AnswerabilityProvider,
        covered_threshold: float,
        partial_threshold: float,
    ) -> None:
        if not isinstance(
            covered_threshold,
            float,
        ):
            raise ValueError(
                "Covered threshold must be a float."
            )

        if not isinstance(
            partial_threshold,
            float,
        ):
            raise ValueError(
                "Partial threshold must be a float."
            )

        if not (
            0.0
            <= partial_threshold
            < covered_threshold
            <= 1.0
        ):
            raise ValueError(
                "Coverage thresholds must satisfy "
                "0.0 <= partial < covered <= 1.0."
            )

        self._answerability_provider = (
            answerability_provider
        )

        self._covered_threshold = (
            covered_threshold
        )

        self._partial_threshold = (
            partial_threshold
        )

    def assess(
        self,
        question: str,
        propositions: MaterialQuestionPropositions,
        evidence_texts: tuple[
            str,
            ...
        ],
    ) -> QuestionEvidenceCoverage:
        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        if not isinstance(
            propositions,
            MaterialQuestionPropositions,
        ):
            raise ValueError(
                "Propositions must be "
                "MaterialQuestionPropositions."
            )

        if not evidence_texts:
            raise ValueError(
                "Evidence texts cannot be empty."
            )

        if not all(
            isinstance(
                evidence_text,
                str,
            )
            and evidence_text.strip()
            for evidence_text
            in evidence_texts
        ):
            raise ValueError(
                "Evidence texts must contain "
                "non-empty strings."
            )

        coverage = tuple(
            self._assess_proposition(
                question,
                proposition,
                evidence_texts,
            )
            for proposition
            in propositions.propositions
        )

        return QuestionEvidenceCoverage(
            propositions=coverage
        )

    def _assess_proposition(
        self,
        question: str,
        proposition: MaterialProposition,
        evidence_texts: tuple[
            str,
            ...
        ],
    ) -> PropositionEvidenceCoverage:
        coverage_query = (
            self._coverage_query(
                question,
                proposition,
            )
        )

        result = score_answerability(
            coverage_query,
            evidence_texts,
            self._answerability_provider,
        )

        strongest_score = (
            result.strongest_score
        )

        if (
            strongest_score
            >= self._covered_threshold
        ):
            status = (
                PropositionCoverageStatus.COVERED
            )

            supporting_indexes = tuple(
                index
                for index, score
                in enumerate(
                    result.scores
                )
                if (
                    score
                    >= self._covered_threshold
                )
            )

        elif (
            strongest_score
            >= self._partial_threshold
        ):
            status = (
                PropositionCoverageStatus.PARTIAL
            )

            supporting_indexes = tuple(
                index
                for index, score
                in enumerate(
                    result.scores
                )
                if (
                    score
                    >= self._partial_threshold
                )
            )

        else:
            status = (
                PropositionCoverageStatus.UNCOVERED
            )

            supporting_indexes = ()

        return PropositionEvidenceCoverage(
            proposition=proposition,
            status=status,
            score=float(
                strongest_score
            ),
            evidence_indexes=(
                supporting_indexes
            ),
        )

    def _coverage_query(
        self,
        question: str,
        proposition: MaterialProposition,
    ) -> str:
        focus_parts: list[str] = []

        self._append_focus(
            focus_parts,
            "subject",
            proposition.subjects,
        )

        self._append_focus(
            focus_parts,
            "scope",
            proposition.scopes,
        )

        self._append_focus(
            focus_parts,
            "relation",
            proposition.relations,
        )

        self._append_focus(
            focus_parts,
            "object",
            proposition.objects,
        )

        self._append_focus(
            focus_parts,
            "requested attribute",
            proposition.requested_attributes,
        )

        self._append_focus(
            focus_parts,
            "qualifier",
            proposition.qualifiers,
        )

        self._append_focus(
            focus_parts,
            "condition",
            proposition.conditions,
        )

        self._append_focus(
            focus_parts,
            "modality",
            proposition.modalities,
        )

        self._append_focus(
            focus_parts,
            "negation",
            proposition.negations,
        )

        if not focus_parts:
            return question

        focus = "; ".join(
            focus_parts
        )

        return (
            f"{question} "
            f"Required proposition: {focus}."
        )

    def _append_focus(
        self,
        parts: list[str],
        label: str,
        requirements: tuple,
    ) -> None:
        if not requirements:
            return

        values = ", ".join(
            requirement.text
            for requirement
            in requirements
        )

        parts.append(
            f"{label}: {values}"
        )