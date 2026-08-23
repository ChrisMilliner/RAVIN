import argparse
from backend.ingestion.acquisition import (
    PolicyLink,
    acquire_policy,
)
from backend.ingestion.processor import (
    process_policy,
)
from backend.retrieval.production import (
    ProductionRetrievalConfig,
    build_production_retrieval_index,
    retrieve_grounded_context,
)
from backend.retrieval.context import (
    ContextAssemblyConfig,
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect production retrieval and "
            "grounded context for candidate "
            "questions."
        )
    )

    parser.add_argument(
        "--question",
        action="append",
        required=True,
        help=(
            "Candidate question to inspect. "
            "Repeat this option to inspect "
            "multiple questions."
        ),
    )

    return parser.parse_args()

def acquire_corpus_chunks():
    all_chunks = []

    print("=== ACQUIRING LIVE POLICIES ===")

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
                f"{policy_id}."
            )

        print(
            policy_id,
            title,
            "->",
            len(ingestion_result.chunks),
            "chunks",
        )

        all_chunks.extend(
            ingestion_result.chunks
        )

    return tuple(all_chunks)

def inspect_questions(
    questions: tuple[str, ...],
) -> None:
    print(
        "=== GROUNDING CANDIDATE "
        "INSPECTION ==="
    )

    print()

    chunks = acquire_corpus_chunks()

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

    embedding_provider = (
        compose_embedding_provider(
            embedding_config,
            provider_factories,
        )
    )

    index = build_production_retrieval_index(
        chunks,
        embedding_provider,
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

    for candidate_number, question in enumerate(
        questions,
        start=1,
    ):
        question = question.strip()

        if not question:
            raise ValueError(
                "Candidate questions must "
                "not be blank."
            )

        candidate_id = (
            f"C{candidate_number:03d}"
        )

        result = retrieve_grounded_context(
            index,
            query=question,
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

        print()
        print("=" * 72)

        print(
            candidate_id,
            "|",
            question,
        )

        print()
        print("--- TOP-5 RETRIEVAL ---")

        for rank, retrieval_result in enumerate(
            result.retrieval_results,
            start=1,
        ):
            chunk = retrieval_result.chunk

            heading = (
                " > ".join(
                    chunk.heading_path
                )
                if chunk.heading_path
                else "(document root)"
            )

            print()
            print(
                "Rank:",
                rank,
            )

            print(
                "Score:",
                f"{retrieval_result.score:.6f}",
            )

            print(
                "Policy:",
                chunk.policy_id,
                "-",
                chunk.policy_title,
            )

            print(
                "Heading:",
                heading,
            )

            print(
                "Chunk:",
                chunk.chunk_index,
            )

            print(
                "Text:",
                chunk.text,
            )

        print()
        print("--- GROUNDED CONTEXT ---")

        print(
            result.rendered_context
        )

    print()
    print("=" * 72)

    print(
        "CANDIDATE INSPECTION COMPLETE"
    )

def main() -> None:
    args = parse_args()

    inspect_questions(
        tuple(args.question)
    )

if __name__ == "__main__":
    main()