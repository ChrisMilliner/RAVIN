"""
Run the configured RAVIN retrieval evaluation from the command line.

This developer entry point loads the evaluation dataset and policy
corpus, builds the configured retrieval stack, executes evaluation, and
prints the resulting retrieval metrics.

Any reported accuracy must be interpreted according to the validation
status of the dataset used for that run.
"""

from backend.evaluation.dataset import (
    load_evaluation_questions,
)
from backend.evaluation.models import EvaluationConfig
from backend.evaluation.runner import (
    run_retrieval_evaluation,
)
from backend.ingestion.acquisition import (
    PolicyLink,
    acquire_policy,
)
from backend.ingestion.processor import process_policy
from backend.retrieval.index import (
    build_semantic_index,
    search_semantic_index,
)
from backend.core.provider_composition import (
    compose_embedding_provider,
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

def main() -> None:
    """Run the preliminary command-line retrieval evaluation workflow.
    """
    print("=== RAVIN RETRIEVAL EVALUATION ===")
    print()

    questions = load_evaluation_questions(
        "evaluation/retrieval_baseline.json"
    )

    print(
        "Evaluation questions:",
        len(questions),
    )

    all_chunks = []

    print()
    print("=== ACQUIRING POLICIES ===")

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
        ingestion_result = process_policy(policy)

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
    print("=== BUILDING SEMANTIC INDEX ===")

    semantic_index = build_semantic_index(
        tuple(all_chunks),
        embedding_provider,
    )

    print(
        "Indexed chunks:",
        len(semantic_index),
    )

    config = EvaluationConfig()

    def retrieve(
        question: str,
        top_k: int,
    ):
        return search_semantic_index(
            semantic_index,
            query=question,
            embedding_provider=embedding_provider,
            top_k=top_k,
        )

    evaluation = run_retrieval_evaluation(
        questions,
        retrieve,
        config,
    )

    print()
    print("=== PER-QUESTION RESULTS ===")

    for question_result in evaluation.question_results:
        print()
        print(
            question_result.question_id,
            "-",
            question_result.question,
        )

        if question_result.first_relevant_rank is None:
            print(
                "First relevant rank: NOT FOUND"
            )
        else:
            print(
                "First relevant rank:",
                question_result.first_relevant_rank,
            )

        for rank, result in enumerate(
            question_result.retrieved_results,
            start=1,
        ):
            print(
                f"  {rank}. "
                f"{result.chunk.policy_title} | "
                f"{' > '.join(result.chunk.heading_path)} | "
                f"{result.score:.4f}"
            )

    print()
    print("=== OVERALL METRICS ===")
    print(
        "Top-1 Accuracy:",
        f"{evaluation.top_1_accuracy:.2%}",
    )
    print(
        f"Hit@{evaluation.top_k}:",
        f"{evaluation.hit_at_k:.2%}",
    )
    print(
        "MRR:",
        f"{evaluation.mrr:.4f}",
    )
    print(
        "Required Top-1:",
        f"{evaluation.pass_threshold:.2%}",
    )
    print(
        "Overall:",
        "PASS" if evaluation.passed else "FAIL",
    )

    print()
    print(
        "NOTE: This is a preliminary demonstration "
        "dataset and is not sufficient to claim "
        "overall system accuracy."
    )

if __name__ == "__main__":
    main()