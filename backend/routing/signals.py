from backend.retrieval.production import (
    GroundedRetrievalResult,
)
from backend.routing.models import (
    EvidenceSignals,
)

def extract_evidence_signals(
    result: GroundedRetrievalResult,
) -> EvidenceSignals:
    retrieved_count = len(
        result.retrieval_results
    )

    context_block_count = (
        result.context.evidence_count
    )

    distinct_policy_count = len(
        {
            block.policy_id
            for block in result.context.blocks
        }
    )

    top_score: float | None = None
    second_score: float | None = None
    score_margin: float | None = None

    if retrieved_count >= 2:
        top_score = (
            result.retrieval_results[0].score
        )

        second_score = (
            result.retrieval_results[1].score
        )

        score_margin = (
            top_score - second_score
        )

    elif retrieved_count == 1:
        top_score = (
            result.retrieval_results[0].score
        )

    return EvidenceSignals(
        retrieved_count=retrieved_count,
        context_block_count=(
            context_block_count
        ),
        distinct_policy_count=(
            distinct_policy_count
        ),
        top_score=top_score,
        second_score=second_score,
        score_margin=score_margin,
    )