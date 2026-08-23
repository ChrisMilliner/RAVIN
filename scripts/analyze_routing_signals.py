from pathlib import Path
from backend.evaluation.dataset import (
    load_evaluation_questions,
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
from backend.routing.signals import (
    extract_evidence_signals,
)
from backend.core.provider_composition import (
    compose_embedding_provider,
    compose_reranker_provider,
)
from backend.core.provider_registry import (
    create_provider_factories,
)
from backend.core.runtime_config_loader import (
    load_runtime_provider_config,
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
    "evaluation/retrieval_baseline.json"
)

def format_score(
    score: float | None,
) -> str:
    if score is None:
        return "N/A"

    return f"{score:.6f}"

def main() -> None:
    print(
        "=== RAVIN ROUTING SIGNAL ANALYSIS ==="
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

        policy = acquire_policy(link)

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

    retrieval_config = (
        ProductionRetrievalConfig()
    )

    context_config = (
        ContextAssemblyConfig()
    )

    questions = load_evaluation_questions(
        DATASET_PATH
    )

    print()
    print(
        "Evaluation questions:",
        len(questions),
    )

    print()
    print("=== SIGNAL RESULTS ===")

    for question in questions:
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

        print()
        print("-" * 72)

        print(
            "Question ID:",
            question.question_id,
        )

        print(
            "Behaviour:",
            question.behavior.value,
        )

        print(
            "Question:",
            question.question,
        )

        print(
            "Retrieved results:",
            signals.retrieved_count,
        )

        print(
            "Context blocks:",
            signals.context_block_count,
        )

        print(
            "Distinct policies:",
            signals.distinct_policy_count,
        )

        print(
            "Top score:",
            format_score(
                signals.top_score
            ),
        )

        print(
            "Second score:",
            format_score(
                signals.second_score
            ),
        )

        print(
            "Score margin:",
            format_score(
                signals.score_margin
            ),
        )

    print()
    print("=" * 72)
    print("SIGNAL ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()