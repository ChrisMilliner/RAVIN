"""
Calculate retrieval-quality metrics and the Top-1 quality gate.

This module provides Top-1 accuracy, Hit@K, mean reciprocal rank, and
quality-gate calculations used by RAVIN retrieval evaluation.

RAVIN's target requires validated Top-1 accuracy of at least 95 percent.
Results from preliminary or development-only datasets must not be
reported as satisfying the formal validated accuracy objective.
"""

def calculate_top_1_accuracy(
    first_relevant_ranks: tuple[int | None, ...],
) -> float:
    if not first_relevant_ranks:
        raise ValueError(
            "Cannot calculate metrics from an empty evaluation set."
        )

    top_1_hits = sum(
        1
        for rank in first_relevant_ranks
        if rank == 1
    )

    return top_1_hits / len(first_relevant_ranks)

def calculate_hit_at_k(
    first_relevant_ranks: tuple[int | None, ...],
    k: int,
) -> float:
    if not first_relevant_ranks:
        raise ValueError(
            "Cannot calculate metrics from an empty evaluation set."
        )

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    hits = sum(
        1
        for rank in first_relevant_ranks
        if rank is not None and rank <= k
    )

    return hits / len(first_relevant_ranks)

def calculate_mrr(
    first_relevant_ranks: tuple[int | None, ...],
) -> float:
    if not first_relevant_ranks:
        raise ValueError(
            "Cannot calculate metrics from an empty evaluation set."
        )

    reciprocal_rank_total = sum(
        1.0 / rank
        for rank in first_relevant_ranks
        if rank is not None
    )

    return (
        reciprocal_rank_total
        / len(first_relevant_ranks)
    )

def meets_top_1_quality_gate(
    top_1_accuracy: float,
    threshold: float,
) -> bool:
    if not 0.0 <= top_1_accuracy <= 1.0:
        raise ValueError(
            "Top-1 accuracy must be between 0 and 1."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Quality threshold must be between 0 and 1."
        )

    return top_1_accuracy >= threshold