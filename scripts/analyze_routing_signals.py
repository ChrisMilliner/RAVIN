"""
Analyse routing evidence signals during RAVIN development.

This developer utility summarizes score distributions and boundary
cases so routing behaviour can be inspected when tuning deterministic
rules or development thresholds.

It is a diagnostic tool and is not part of the production request path.
"""

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from backend.core.provider_composition import (
    compose_answerability_provider,
    compose_embedding_provider,
    compose_reranker_provider,
)
from backend.core.provider_registry import (
    create_provider_factories,
)
from backend.core.runtime_config_loader import (
    load_runtime_provider_config,
)
from backend.evaluation.routing_dataset import (
    load_routing_evaluation_questions,
)
from backend.ingestion.acquisition import (
    PolicyLink,
    acquire_policy,
)
from backend.ingestion.processor import (
    process_policy,
)
from backend.retrieval.context import (
    ContextAssemblyConfig,
)
from backend.retrieval.production import (
    ProductionRetrievalConfig,
    build_production_retrieval_index,
    retrieve_grounded_context,
)
from backend.routing.answerability import (
    score_answerability,
)
from backend.routing.models import (
    EvidenceSufficiency,
)
from backend.routing.signals import (
    extract_evidence_signals,
)

POLICIES = (
    ("208", "Academic Dress Policy"),
    ("220", "Academic Progression Review Policy"),
    ("76", "Academic Promotions Policy"),
    ("420", "Academic Staff Qualifications Policy"),
    ("169", "Admissions Policy"),
    ("340", "Admissions Procedure"),
)
DATASET_PATH = Path(
    "evaluation/routing_baseline.json"
)

@dataclass(frozen=True)
class SignalObservation:
    """Record retrieval and answerability signals observed for one development question.
    """

    question_id: str
    question: str
    expected_intent: str
    expected_sufficiency: EvidenceSufficiency
    top_score: float | None
    second_score: float | None
    score_margin: float | None
    context_block_count: int
    distinct_policy_count: int
    answerability_scores: tuple[
        float,
        ...
    ]
    strongest_answerability: float | None

def format_score(
    score: float | None,
) -> str:
    """Format an optional diagnostic score for console output.
    """
    if score is None:
        return "N/A"

    return f"{score:.6f}"

def format_distribution(
    values: tuple[
        float | int,
        ...
    ],
) -> str:
    """Format minimum, median, and maximum diagnostic values.
    """
    if not values:
        return "N/A"

    return (
        f"min={min(values):.6f}, "
        f"median={median(values):.6f}, "
        f"max={max(values):.6f}"
    )

def format_heading(
    heading_path: tuple[
        str,
        ...
    ],
) -> str:
    """Format a policy heading path for diagnostic output.
    """
    if not heading_path:
        return "(document root)"

    return " > ".join(
        heading_path
    )

def print_observation_summary(
    observations: tuple[
        SignalObservation,
        ...
    ],
) -> None:
    """Print grouped distributions for measured evidence-sufficiency signals.
    """
    print()
    print("=" * 72)
    print("=== EXPECTED SUFFICIENCY DISTRIBUTIONS ===")

    for expected in (
        EvidenceSufficiency.SUFFICIENT,
        EvidenceSufficiency.INSUFFICIENT,
    ):
        group = tuple(
            observation
            for observation in observations
            if (
                observation.expected_sufficiency
                == expected
            )
        )

        top_scores = tuple(
            observation.top_score
            for observation in group
            if observation.top_score is not None
        )

        margins = tuple(
            observation.score_margin
            for observation in group
            if observation.score_margin is not None
        )

        answerability_scores = tuple(
            observation.strongest_answerability
            for observation in group
            if (
                observation.strongest_answerability
                is not None
            )
        )

        context_counts = tuple(
            observation.context_block_count
            for observation in group
        )

        policy_counts = tuple(
            observation.distinct_policy_count
            for observation in group
        )

        print()
        print(
            expected.value.upper(),
            f"({len(group)} questions)",
        )

        print(
            "Top retrieval score:",
            format_distribution(
                top_scores
            ),
        )

        print(
            "Retrieval score margin:",
            format_distribution(
                margins
            ),
        )

        print(
            "Strongest answerability:",
            format_distribution(
                answerability_scores
            ),
        )

        print(
            "Context block count:",
            format_distribution(
                context_counts
            ),
        )

        print(
            "Distinct policy count:",
            format_distribution(
                policy_counts
            ),
        )

def print_boundary_cases(
    observations: tuple[
        SignalObservation,
        ...
    ],
) -> None:
    """Print high-risk sufficient and insufficient diagnostic boundary cases.
    """
    sufficient = tuple(
        observation
        for observation in observations
        if (
            observation.expected_sufficiency
            == EvidenceSufficiency.SUFFICIENT
        )
    )

    insufficient = tuple(
        observation
        for observation in observations
        if (
            observation.expected_sufficiency
            == EvidenceSufficiency.INSUFFICIENT
        )
    )

    sufficient_by_answerability = sorted(
        sufficient,
        key=lambda observation: (
            observation.strongest_answerability
            if (
                observation.strongest_answerability
                is not None
            )
            else float("inf")
        ),
    )

    insufficient_by_answerability = sorted(
        insufficient,
        key=lambda observation: (
            observation.strongest_answerability
            if (
                observation.strongest_answerability
                is not None
            )
            else float("-inf")
        ),
        reverse=True,
    )

    sufficient_by_retrieval = sorted(
        sufficient,
        key=lambda observation: (
            observation.top_score
            if observation.top_score is not None
            else float("inf")
        ),
    )

    insufficient_by_retrieval = sorted(
        insufficient,
        key=lambda observation: (
            observation.top_score
            if observation.top_score is not None
            else float("-inf")
        ),
        reverse=True,
    )

    print()
    print("=" * 72)
    print("=== ANSWERABILITY BOUNDARY CASES ===")

    print()
    print(
        "Highest answerability among expected "
        "INSUFFICIENT:"
    )

    for observation in (
        insufficient_by_answerability[:5]
    ):
        print(
            observation.question_id,
            "|",
            format_score(
                observation.strongest_answerability
            ),
            "|",
            observation.question,
        )

    print()
    print(
        "Lowest answerability among expected "
        "SUFFICIENT:"
    )

    for observation in (
        sufficient_by_answerability[:5]
    ):
        print(
            observation.question_id,
            "|",
            format_score(
                observation.strongest_answerability
            ),
            "|",
            observation.question,
        )

    print()
    print("=" * 72)
    print("=== RETRIEVAL BOUNDARY CASES ===")

    print()
    print(
        "Highest retrieval score among expected "
        "INSUFFICIENT:"
    )

    for observation in (
        insufficient_by_retrieval[:5]
    ):
        print(
            observation.question_id,
            "|",
            format_score(
                observation.top_score
            ),
            "|",
            observation.question,
        )

    print()
    print(
        "Lowest retrieval score among expected "
        "SUFFICIENT:"
    )

    for observation in (
        sufficient_by_retrieval[:5]
    ):
        print(
            observation.question_id,
            "|",
            format_score(
                observation.top_score
            ),
            "|",
            observation.question,
        )

def main() -> None:
    """Acquire development policies and report routing signals without applying gates.
    """
    print(
        "=== RAVIN EVIDENCE SUFFICIENCY "
        "SIGNAL ANALYSIS ==="
    )

    print()
    print(
        "Measurement only - no sufficiency "
        "thresholds or predictions are applied."
    )

    print()
    print("=== ACQUIRING LIVE POLICIES ===")

    all_chunks = []

    for policy_id, title in POLICIES:
        link = PolicyLink(
            policy_id=policy_id,
            title=title,
            url=(
                "https://policies.latrobe.edu.au/"
                f"document/view.php?id={policy_id}"
            ),
        )

        policy = acquire_policy(
            link
        )

        ingestion_result = process_policy(
            policy
        )

        if not ingestion_result.chunks:
            raise RuntimeError(
                "Policy ingestion failed for "
                f"{policy_id}: "
                f"{ingestion_result.error}"
            )

        print(
            policy.policy_id,
            policy.title,
            "->",
            len(ingestion_result.chunks),
            "chunks",
        )

        all_chunks.extend(
            ingestion_result.chunks
        )

    print()
    print(
        "Total policy chunks:",
        len(all_chunks),
    )

    runtime_provider_config = (
        load_runtime_provider_config()
    )

    provider_factories = (
        create_provider_factories()
    )

    embedding_config = (
        runtime_provider_config
        .retrieval
        .embedding
    )

    reranker_config = (
        runtime_provider_config
        .retrieval
        .reranker
    )

    answerability_config = (
        runtime_provider_config
        .answerability
    )

    print()
    print("=== EMBEDDING PROVIDER ===")

    print(
        "Provider:",
        embedding_config.provider,
    )

    print(
        "Model:",
        embedding_config.model,
    )

    embedding_provider = (
        compose_embedding_provider(
            embedding_config,
            provider_factories,
        )
    )

    print()
    print("=== BUILDING PRODUCTION INDEX ===")

    index = build_production_retrieval_index(
        tuple(all_chunks),
        embedding_provider,
    )

    print(
        "Indexed chunks:",
        len(index),
    )

    print()
    print("=== RERANKER PROVIDER ===")

    print(
        "Provider:",
        reranker_config.provider,
    )

    print(
        "Model:",
        reranker_config.model,
    )

    reranker_provider = (
        compose_reranker_provider(
            reranker_config,
            provider_factories,
        )
    )

    print()
    print("=== ANSWERABILITY PROVIDER ===")

    print(
        "Provider:",
        answerability_config.provider,
    )

    print(
        "Model:",
        answerability_config.model,
    )

    answerability_provider = (
        compose_answerability_provider(
            answerability_config,
            provider_factories,
        )
    )

    retrieval_config = (
        ProductionRetrievalConfig()
    )

    context_config = (
        ContextAssemblyConfig()
    )

    all_questions = (
        load_routing_evaluation_questions(
            DATASET_PATH
        )
    )

    clear_questions = tuple(
        question
        for question in all_questions
        if (
            question.expected_sufficiency
            is not None
        )
    )

    ambiguous_count = (
        len(all_questions)
        - len(clear_questions)
    )

    print()
    print("=== ROUTING DATASET ===")

    print(
        "Dataset:",
        DATASET_PATH,
    )

    print(
        "Total questions:",
        len(all_questions),
    )

    print(
        "Clear sufficiency questions:",
        len(clear_questions),
    )

    print(
        "Ambiguous questions excluded:",
        ambiguous_count,
    )

    observations: list[
        SignalObservation
    ] = []

    print()
    print("=== SIGNAL RESULTS ===")

    for question in clear_questions:
        expected_sufficiency = (
            question.expected_sufficiency
        )

        if expected_sufficiency is None:
            raise RuntimeError(
                "Clear sufficiency analysis "
                "received an ambiguous question."
            )

        result = retrieve_grounded_context(
            index,
            query=question.question,
            embedding_provider=(
                embedding_provider
            ),
            reranker_provider=(
                reranker_provider
            ),
            retrieval_config=(
                retrieval_config
            ),
            context_config=(
                context_config
            ),
        )

        signals = extract_evidence_signals(
            result
        )

        evidence_texts = tuple(
            block.text
            for block in result.context.blocks
        )

        answerability_scores: tuple[
            float,
            ...
        ] = ()

        strongest_answerability: (
            float | None
        ) = None

        if evidence_texts:
            answerability_result = (
                score_answerability(
                    question.question,
                    evidence_texts,
                    answerability_provider,
                )
            )

            answerability_scores = (
                answerability_result.scores
            )

            strongest_answerability = (
                answerability_result
                .strongest_score
            )

        policy_ids = tuple(
            dict.fromkeys(
                block.policy_id
                for block
                in result.context.blocks
            )
        )

        print()
        print("-" * 72)

        print(
            "Question ID:",
            question.question_id,
        )

        print(
            "Expected intent:",
            question.expected_intent.value,
        )

        print(
            "Expected sufficiency:",
            expected_sufficiency.value,
        )

        print(
            "Question:",
            question.question,
        )

        print()
        print("Retrieval:")

        print(
            "  Retrieved results:",
            signals.retrieved_count,
        )

        print(
            "  Context blocks:",
            signals.context_block_count,
        )

        print(
            "  Distinct policies:",
            signals.distinct_policy_count,
        )

        print(
            "  Policy IDs:",
            (
                ", ".join(policy_ids)
                if policy_ids
                else "N/A"
            ),
        )

        print(
            "  Top score:",
            format_score(
                signals.top_score
            ),
        )

        print(
            "  Second score:",
            format_score(
                signals.second_score
            ),
        )

        print(
            "  Score margin:",
            format_score(
                signals.score_margin
            ),
        )

        print()
        print("Answerability:")

        if answerability_scores:
            for (
                position,
                (
                    block,
                    score,
                ),
            ) in enumerate(
                zip(
                    result.context.blocks,
                    answerability_scores,
                ),
                start=1,
            ):
                print(
                    f"  E{position}: "
                    f"{score:.6f} "
                    f"| Policy {block.policy_id} "
                    f"| {block.policy_title} "
                    f"| "
                    f"{format_heading(block.heading_path)}"
                )

        else:
            print(
                "  No grounded evidence blocks."
            )

        print(
            "  Strongest:",
            format_score(
                strongest_answerability
            ),
        )

        observations.append(
            SignalObservation(
                question_id=(
                    question.question_id
                ),
                question=question.question,
                expected_intent=(
                    question.expected_intent.value
                ),
                expected_sufficiency=(
                    expected_sufficiency
                ),
                top_score=signals.top_score,
                second_score=(
                    signals.second_score
                ),
                score_margin=(
                    signals.score_margin
                ),
                context_block_count=(
                    signals.context_block_count
                ),
                distinct_policy_count=(
                    signals.distinct_policy_count
                ),
                answerability_scores=(
                    answerability_scores
                ),
                strongest_answerability=(
                    strongest_answerability
                ),
            )
        )

    observations_tuple = tuple(
        observations
    )

    print_observation_summary(
        observations_tuple
    )

    print_boundary_cases(
        observations_tuple
    )

    print()
    print("=" * 72)

    print(
        "SIGNAL ANALYSIS COMPLETE"
    )

    print(
        "No sufficiency thresholds or "
        "predictions were applied."
    )

if __name__ == "__main__":
    main()