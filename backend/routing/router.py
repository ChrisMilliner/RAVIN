"""
Select the final RAVIN answer behaviour from assessed control state.

Routing combines question intent and evidence sufficiency to select
DIRECT_ANSWER, GROUNDED_OVERVIEW, CLARIFY, or NO_GROUNDED_ANSWER.

The mapping is deterministic. Ambiguous questions route to
clarification, while clear questions with insufficient evidence route
to a controlled no-grounded-answer response.
"""

from backend.behavior import AnswerBehavior
from backend.routing.models import (
    EvidenceAssessment,
    EvidenceSufficiency,
    QuestionAssessment,
    QuestionIntent,
    RoutingResult,
)

def route_answer(
    question_assessment: QuestionAssessment,
    evidence_assessment: EvidenceAssessment,
) -> RoutingResult:
    """Map deterministic question and evidence assessments to answer behavior.

    Ambiguous questions clarify, unsupported clear questions fail closed,
    focused supported questions answer directly, and broad supported
    questions produce grounded overviews.
    """
    if (
        question_assessment.intent
        == QuestionIntent.AMBIGUOUS
    ):
        return RoutingResult(
            behavior=AnswerBehavior.CLARIFY,
            question_assessment=(
                question_assessment
            ),
            evidence_assessment=(
                evidence_assessment
            ),
            reason=(
                "The question is ambiguous and "
                "requires clarification before "
                "an answer can be grounded."
            ),
        )

    if (
        evidence_assessment.sufficiency
        != EvidenceSufficiency.SUFFICIENT
    ):
        return RoutingResult(
            behavior=(
                AnswerBehavior.NO_GROUNDED_ANSWER
            ),
            question_assessment=(
                question_assessment
            ),
            evidence_assessment=(
                evidence_assessment
            ),
            reason=(
                "The question is clear, but "
                "sufficient grounded evidence "
                "was not established."
            ),
        )

    if (
        question_assessment.intent
        == QuestionIntent.FOCUSED
    ):
        return RoutingResult(
            behavior=(
                AnswerBehavior.DIRECT_ANSWER
            ),
            question_assessment=(
                question_assessment
            ),
            evidence_assessment=(
                evidence_assessment
            ),
            reason=(
                "The question is focused and "
                "sufficient grounded evidence "
                "is available."
            ),
        )

    if (
        question_assessment.intent
        == QuestionIntent.BROAD
    ):
        return RoutingResult(
            behavior=(
                AnswerBehavior.GROUNDED_OVERVIEW
            ),
            question_assessment=(
                question_assessment
            ),
            evidence_assessment=(
                evidence_assessment
            ),
            reason=(
                "The question is broad and "
                "sufficient grounded evidence "
                "is available."
            ),
        )

    raise RuntimeError(
        "Unsupported question intent."
    )