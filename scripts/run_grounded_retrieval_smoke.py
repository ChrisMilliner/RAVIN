from pathlib import Path
from backend.evaluation.dataset import (
    load_evaluation_questions,
)
from backend.ingestion.acquisition import (
    PolicyLink,
    acquire_policy,
)
from backend.ingestion.processor import process_policy
from backend.retrieval.context import (
    ContextAssemblyConfig,
)
from backend.retrieval.cross_encoder_provider import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderRerankerProvider,
)
from backend.retrieval.production import (
    ProductionRetrievalConfig,
    build_production_retrieval_index,
    retrieve_grounded_context,
)
from backend.retrieval.sentence_transformer_provider import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbeddingProvider,
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
SMOKE_QUESTION_IDS = (
    "RB017",
    "RB002",
    "RB025",
)

def main() -> None:
    print("=== RAVIN GROUNDED RETRIEVAL SMOKE TEST ===")

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

    print()
    print("=== LOADING EMBEDDING MODEL ===")
    print(
        "Embedding model:",
        DEFAULT_EMBEDDING_MODEL,
    )

    embedding_provider = (
        SentenceTransformerEmbeddingProvider()
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
    print("=== LOADING RERANKER ===")
    print(
        "Reranker model:",
        DEFAULT_RERANKER_MODEL,
    )

    reranker_provider = (
        CrossEncoderRerankerProvider()
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

    questions_by_id = {
        question.question_id: question
        for question in questions
    }

    for question_id in SMOKE_QUESTION_IDS:
        question = questions_by_id.get(
            question_id
        )

        if question is None:
            raise RuntimeError(
                "Smoke question was not found: "
                f"{question_id}"
            )

        print()
        print("=" * 72)
        print(
            "QUESTION:",
            question_id,
        )
        print(
            question.question
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
                f"Rank {rank}"
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
        print("--- CONTEXT SELECTION ---")
        print(
            "Retrieved seeds:",
            len(
                result.retrieval_results
            ),
        )
        print(
            "Selected context chunks:",
            len(
                result.context_chunks
            ),
        )
        print(
            "Grounded evidence blocks:",
            result.context.evidence_count,
        )

        print()
        print("--- GROUNDED BLOCKS ---")

        for evidence_number, block in enumerate(
            result.context.blocks,
            start=1,
        ):
            heading = (
                " > ".join(
                    block.heading_path
                )
                if block.heading_path
                else "(document root)"
            )

            print()
            print(
                f"[E{evidence_number}]"
            )
            print(
                "Policy:",
                block.policy_id,
                "-",
                block.policy_title,
            )
            print(
                "Heading:",
                heading,
            )
            print(
                "Chunk range:",
                (
                    str(
                        block.start_chunk_index
                    )
                    if (
                        block.start_chunk_index
                        == block.end_chunk_index
                    )
                    else (
                        f"{block.start_chunk_index}-"
                        f"{block.end_chunk_index}"
                    )
                ),
            )
            print(
                "Source:",
                block.source_url,
            )
            print(
                "Text:",
                block.text,
            )

        print()
        print("--- EXACT LLM-FACING CONTEXT ---")
        print(
            result.rendered_context
        )

    print()
    print("=" * 72)
    print("SMOKE TEST COMPLETE")


if __name__ == "__main__":
    main()