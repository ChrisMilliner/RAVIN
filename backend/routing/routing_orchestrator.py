"""
Coordinate deterministic question assessment and behaviour routing.

The routing orchestrator combines intent classification and
evidence-sufficiency assessment into the QuestionAssessment consumed by
the final router.

It coordinates control components only and does not generate user-facing
answer text.
"""

from backend.retrieval.production import (
    GroundedRetrievalResult,
)
from backend.routing.evidence_assessor import (
    EvidenceSufficiencyAssessor,
    assess_evidence_sufficiency,
)
from backend.routing.intent_classifier import (
    QuestionIntentClassifier,
    classify_question_intent,
)
from backend.routing.models import (
    EvidenceAssessment,
    EvidenceSignals,
    EvidenceSufficiency,
    QuestionAssessment,
    QuestionIntent,
    RoutingResult,
)
from backend.routing.router import (
    route_answer,
)

def orchestrate_answer_routing(
    question: str,
    retrieval_result: (
        GroundedRetrievalResult | None
    ),
    intent_classifier: QuestionIntentClassifier,
    evidence_assessor: EvidenceSufficiencyAssessor,
) -> RoutingResult:
    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    intent = classify_question_intent(
        question,
        intent_classifier,
    )

    question_assessment = (
        QuestionAssessment(
            intent=intent,
            reason=(
                _question_assessment_reason(
                    intent
                )
            ),
        )
    )

    if intent == QuestionIntent.AMBIGUOUS:
        return route_answer(
            question_assessment,
            _skipped_evidence_assessment(),
        )

    if retrieval_result is None:
        raise ValueError(
            "A retrieval result is required "
            "for a clear question."
        )

    evidence_assessment = (
        assess_evidence_sufficiency(
            question,
            intent,
            retrieval_result,
            evidence_assessor,
        )
    )

    return route_answer(
        question_assessment,
        evidence_assessment,
    )

def _question_assessment_reason(
    intent: QuestionIntent,
) -> str:
    if intent == QuestionIntent.AMBIGUOUS:
        return (
            "The question requires clarification "
            "before evidence can be assessed."
        )

    if intent == QuestionIntent.BROAD:
        return (
            "The question requests a broad "
            "grounded overview."
        )

    if intent == QuestionIntent.FOCUSED:
        return (
            "The question requests a focused "
            "grounded answer."
        )

    raise RuntimeError(
        "Unsupported question intent."
    )

def _skipped_evidence_assessment(
) -> EvidenceAssessment:
    return EvidenceAssessment(
        sufficiency=(
            EvidenceSufficiency.UNCERTAIN
        ),
        signals=EvidenceSignals(
            retrieved_count=0,
            context_block_count=0,
            distinct_policy_count=0,
            top_score=None,
            second_score=None,
            score_margin=None,
        ),
        reason=(
            "Evidence assessment was skipped "
            "because the question requires "
            "clarification."
        ),
    )