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