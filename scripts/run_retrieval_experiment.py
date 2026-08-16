from datetime import datetime, timezone
from backend.evaluation.dataset import (
    load_evaluation_questions,
)
from backend.evaluation.experiment_models import (
    DatasetValidationStatus,
    ExperimentSelectionDecision,
    RetrievalExperimentConfig,
)
from backend.evaluation.experiments import (
    compare_retrieval_experiments,
)
from backend.evaluation.models import (
    EvaluationConfig,
    EvaluationRunResult,
)
from backend.evaluation.runner import (
    run_retrieval_evaluation,
)
from backend.ingestion.acquisition import (
    PolicyLink,
    acquire_policy,
)
from backend.ingestion.processor import process_policy
from backend.retrieval.hybrid import (
    search_hybrid_index,
)
from backend.retrieval.cross_encoder_provider import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderRerankerProvider,
)
from backend.retrieval.index import (
    BODY_ONLY_EMBEDDING,
    RETRIEVAL_TEXT_EMBEDDING,
    TITLE_BODY_EMBEDDING,
    build_semantic_index,
    search_semantic_index,
)
from backend.retrieval.reranking import (
    rerank_results,
)
from backend.retrieval.sentence_transformer_provider import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbeddingProvider,
)
from backend.evaluation.reporting import (
    build_experiment_record,
    calculate_corpus_sha256,
    calculate_file_sha256,
    ensure_clean_git_working_tree,
    get_repository_commit,
    write_experiment_record,
)

POLICIES = (
    ("208", "Academic Dress Policy"),
    ("220", "Academic Progression Review Policy"),
    ("76", "Academic Promotions Policy"),
    ("420", "Academic Staff Qualifications Policy"),
    ("169", "Admissions Policy"),
    ("340", "Admissions Procedure"),
)
DATASET_PATH = (
    "evaluation/retrieval_baseline.json"
)
EXPERIMENT_OUTPUT_DIRECTORY = (
    "evaluation/experiments"
)
SEMANTIC_WEIGHT = 0.85
LEXICAL_WEIGHT = 0.15
BASELINE_RERANK_DEPTH = 5
CANDIDATE_RERANK_DEPTH = 11

def format_rank(
    rank: int | None,
) -> str:
    if rank is None:
        return "NOT FOUND"

    return str(rank)

def rank_value(
    rank: int | None,
    top_k: int,
) -> int:
    if rank is None:
        return top_k + 1

    return rank

def print_evaluation_metrics(
    name: str,
    evaluation: EvaluationRunResult,
) -> None:
    print()
    print(f"=== {name} ===")
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

def main() -> None:
    ensure_clean_git_working_tree()

    repository_commit = (
        get_repository_commit()
    )

    print(
        "=== RAVIN RETRIEVAL EXPERIMENT ==="
    )
    print()

    questions = load_evaluation_questions(
        DATASET_PATH
    )

    dataset_sha256 = (
        calculate_file_sha256(
            DATASET_PATH
        )
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
        ingestion_result = process_policy(
            policy
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
    print(
        "=== BUILDING RETRIEVAL-TEXT "
        "SEMANTIC INDEX ==="
    )

    embedding_provider = (
        SentenceTransformerEmbeddingProvider()
    )

    retrieval_text_index = (
        build_semantic_index(
            tuple(all_chunks),
            embedding_provider,
            embedding_text_strategy=(
                RETRIEVAL_TEXT_EMBEDDING
            ),
        )
    )

    print(
        "Retrieval-text indexed chunks:",
        len(retrieval_text_index),
    )

    print()
    print(
        "=== BUILDING BODY-ONLY "
        "SEMANTIC INDEX ==="
    )

    body_only_index = (
        build_semantic_index(
            tuple(all_chunks),
            embedding_provider,
            embedding_text_strategy=(
                BODY_ONLY_EMBEDDING
            ),
        )
    )

    print(
        "Body-only indexed chunks:",
        len(body_only_index),
    )

    print()
    print(
        "=== BUILDING TITLE-BODY "
        "SEMANTIC INDEX ==="
    )

    title_body_index = (
        build_semantic_index(
            tuple(all_chunks),
            embedding_provider,
            embedding_text_strategy=(
                TITLE_BODY_EMBEDDING
            ),
        )
    )

    print(
        "Title-body indexed chunks:",
        len(title_body_index),
    )

    retrieval_text_corpus_sha256 = (
        calculate_corpus_sha256(
            retrieval_text_index
        )
    )

    body_only_corpus_sha256 = (
        calculate_corpus_sha256(
            body_only_index
        )
    )

    title_body_corpus_sha256 = (
        calculate_corpus_sha256(
            title_body_index
        )
    )

    if not (
        retrieval_text_corpus_sha256
        == body_only_corpus_sha256
        == title_body_corpus_sha256
    ):
        raise RuntimeError(
            "Embedding strategy changed the "
            "underlying retrieval corpus."
        )

    corpus_sha256 = (
        retrieval_text_corpus_sha256
    )

    evaluation_config = EvaluationConfig()

    print()
    print(
        "=== LOADING CROSS-ENCODER RERANKER ==="
    )
    print(
        "Reranker model:",
        DEFAULT_RERANKER_MODEL,
    )

    reranker_provider = (
        CrossEncoderRerankerProvider()
    )

    print(
        "Cross-encoder reranker loaded."
    )

    def retrieve_baseline(
        question: str,
        top_k: int,
    ):
        return search_semantic_index(
            retrieval_text_index,
            query=question,
            embedding_provider=(
                embedding_provider
            ),
            top_k=top_k,
        )

    def retrieve_candidate(
        question: str,
        top_k: int,
    ):
        return search_hybrid_index(
            retrieval_text_index,
            query=question,
            embedding_provider=(
                embedding_provider
            ),
            top_k=top_k,
            semantic_weight=SEMANTIC_WEIGHT,
            lexical_weight=LEXICAL_WEIGHT,
        )

    def retrieve_candidate_v2(
        question: str,
        top_k: int,
    ):
        hybrid_results = search_hybrid_index(
            retrieval_text_index,
            query=question,
            embedding_provider=(
                embedding_provider
            ),
            top_k=top_k,
            semantic_weight=SEMANTIC_WEIGHT,
            lexical_weight=LEXICAL_WEIGHT,
        )

        return rerank_results(
            query=question,
            results=hybrid_results,
            reranker_provider=(
                reranker_provider
            ),
        )

    def retrieve_candidate_v3(
        question: str,
        top_k: int,
    ):
        hybrid_results = search_hybrid_index(
            body_only_index,
            query=question,
            embedding_provider=(
                embedding_provider
            ),
            top_k=BASELINE_RERANK_DEPTH,
            semantic_weight=SEMANTIC_WEIGHT,
            lexical_weight=LEXICAL_WEIGHT,
        )

        reranked_results = rerank_results(
            query=question,
            results=hybrid_results,
            reranker_provider=(
                reranker_provider
            ),
        )

        return reranked_results[:top_k]

    def retrieve_candidate_v5(
        question: str,
        top_k: int,
    ):
        hybrid_results = search_hybrid_index(
            body_only_index,
            query=question,
            embedding_provider=(
                embedding_provider
            ),
            top_k=CANDIDATE_RERANK_DEPTH,
            semantic_weight=SEMANTIC_WEIGHT,
            lexical_weight=LEXICAL_WEIGHT,
        )

        reranked_results = rerank_results(
            query=question,
            results=hybrid_results,
            reranker_provider=(
                reranker_provider
            ),
        )

        return reranked_results[:top_k]

    def retrieve_candidate_v4(
        question: str,
        top_k: int,
    ):
        hybrid_results = search_hybrid_index(
            title_body_index,
            query=question,
            embedding_provider=(
                embedding_provider
            ),
            top_k=top_k,
            semantic_weight=SEMANTIC_WEIGHT,
            lexical_weight=LEXICAL_WEIGHT,
        )

        return rerank_results(
            query=question,
            results=hybrid_results,
            reranker_provider=(
                reranker_provider
            ),
        )

    print()
    print(
        "=== RUNNING SEMANTIC BASELINE ==="
    )

    baseline = run_retrieval_evaluation(
        questions,
        retrieve_baseline,
        evaluation_config,
    )

    print(
        "Baseline evaluation complete."
    )

    print()
    print(
        "=== RUNNING HYBRID CANDIDATE V1 ==="
    )
    print(
        "Semantic weight:",
        f"{SEMANTIC_WEIGHT:.0%}",
    )
    print(
        "Lexical weight:",
        f"{LEXICAL_WEIGHT:.0%}",
    )

    candidate_v1 = run_retrieval_evaluation(
        questions,
        retrieve_candidate,
        evaluation_config,
    )

    print(
        "Candidate v1 evaluation complete."
    )

    print()
    print(
        "=== RUNNING RERANKED CANDIDATE V2 ==="
    )
    print(
        "Hybrid retrieval depth:",
        evaluation_config.top_k,
    )
    print(
        "Reranker model:",
        DEFAULT_RERANKER_MODEL,
    )

    candidate_v2 = run_retrieval_evaluation(
        questions,
        retrieve_candidate_v2,
        evaluation_config,
    )

    print(
        "Candidate v2 evaluation complete."
    )

    print()
    print(
        "=== RUNNING BODY-ONLY "
        "EMBEDDING CANDIDATE V3 ==="
    )
    print(
        "Embedding text strategy:",
        BODY_ONLY_EMBEDDING,
    )
    print(
        "Hybrid retrieval depth:",
        evaluation_config.top_k,
    )
    print(
        "Reranker model:",
        DEFAULT_RERANKER_MODEL,
    )

    candidate_v3 = run_retrieval_evaluation(
        questions,
        retrieve_candidate_v3,
        evaluation_config,
    )

    print(
        "Candidate v3 evaluation complete."
    )

    print()
    print(
        "=== RUNNING TITLE-BODY "
        "EMBEDDING CANDIDATE V4 ==="
    )
    print(
        "Embedding text strategy:",
        TITLE_BODY_EMBEDDING,
    )
    print(
        "Hybrid retrieval depth:",
        evaluation_config.top_k,
    )
    print(
        "Reranker model:",
        DEFAULT_RERANKER_MODEL,
    )

    candidate_v4 = run_retrieval_evaluation(
        questions,
        retrieve_candidate_v4,
        evaluation_config,
    )

    print(
        "Candidate v4 evaluation complete."
    )

    print()
    print(
        "=== RUNNING INCREASED RERANK "
        "DEPTH CANDIDATE V5 ==="
    )
    print(
        "Embedding text strategy:",
        BODY_ONLY_EMBEDDING,
    )
    print(
        "Baseline rerank depth:",
        BASELINE_RERANK_DEPTH,
    )
    print(
        "Candidate rerank depth:",
        CANDIDATE_RERANK_DEPTH,
    )
    print(
        "Reranker model:",
        DEFAULT_RERANKER_MODEL,
    )

    candidate_v5 = run_retrieval_evaluation(
        questions,
        retrieve_candidate_v5,
        evaluation_config,
    )

    print(
        "Candidate v5 evaluation complete."
    )

    experiment_config = (
        RetrievalExperimentConfig(
            experiment_name=(
                "Cross-encoder rerank depth 11 v5"
            ),
            baseline_name=(
                "Body-only MiniLM embeddings + "
                "Hybrid 85/15 + MS MARCO "
                "MiniLM-L6-v2 reranking "
                "depth 5 v3"
            ),
            candidate_name=(
                "Body-only MiniLM embeddings + "
                "Hybrid 85/15 + MS MARCO "
                "MiniLM-L6-v2 reranking "
                "depth 11 v5"
            ),
            dataset_name=(
                "RAVIN Preliminary Retrieval "
                "Development Baseline v1.2"
            ),
            dataset_status=(
                DatasetValidationStatus.PRELIMINARY
            ),
            top_k=evaluation_config.top_k,
            quality_threshold=(
                evaluation_config
                .top_1_pass_threshold
            ),
        )
    )

    comparison = (
        compare_retrieval_experiments(
            candidate_v3,
            candidate_v5,
            experiment_config,
        )
    )

    print_evaluation_metrics(
        "BASELINE V3 - BODY-ONLY + "
        "RERANK DEPTH 5",
        candidate_v3,
    )

    print_evaluation_metrics(
        "CANDIDATE V5 - BODY-ONLY + "
        "RERANK DEPTH 11",
        candidate_v5,
    )

    print()
    print(
        "=== V3 -> V5 METRIC COMPARISON ==="
    )

    print(
        "Top-1 delta:",
        f"{comparison.top_1.delta:+.2%}",
    )
    print(
        f"Hit@{evaluation_config.top_k} delta:",
        f"{comparison.hit_at_k.delta:+.2%}",
    )
    print(
        "MRR delta:",
        f"{comparison.mrr.delta:+.4f}",
    )

    improved = []
    regressed = []
    unchanged = []

    for change in (
        comparison.question_rank_changes
    ):
        baseline_rank_value = rank_value(
            change.baseline_rank,
            evaluation_config.top_k,
        )

        candidate_rank_value = rank_value(
            change.candidate_rank,
            evaluation_config.top_k,
        )

        if (
            candidate_rank_value
            < baseline_rank_value
        ):
            improved.append(change)
        elif (
            candidate_rank_value
            > baseline_rank_value
        ):
            regressed.append(change)
        else:
            unchanged.append(change)

    print()
    print("=== IMPROVED QUESTIONS ===")

    if not improved:
        print("None")
    else:
        for change in improved:
            print(
                f"{change.question_id}: "
                f"{format_rank(change.baseline_rank)}"
                " -> "
                f"{format_rank(change.candidate_rank)}"
            )

    print()
    print("=== REGRESSED QUESTIONS ===")

    if not regressed:
        print("None")
    else:
        for change in regressed:
            print(
                f"{change.question_id}: "
                f"{format_rank(change.baseline_rank)}"
                " -> "
                f"{format_rank(change.candidate_rank)}"
            )

    print()
    print("=== UNCHANGED QUESTIONS ===")
    print(
        "Count:",
        len(unchanged),
    )

    print()
    print("=== QUALITY DECISION ===")

    print(
        "Relative performance:",
        comparison.direction.value.upper(),
    )

    print(
        "Top-1 quality gate:",
        (
            "PASS"
            if comparison.quality_gate_passed
            else "FAIL"
        ),
    )

    print(
        "Required Top-1:",
        f"{experiment_config.quality_threshold:.2%}",
    )

    print(
        "Dataset validation status:",
        experiment_config.dataset_status.value,
    )

    print(
        "Validated dataset gate:",
        (
            "PASS"
            if (
                comparison
                .validated_dataset_gate_passed
            )
            else "FAIL"
        ),
    )

    print(
        "Selection decision:",
        (
            comparison
            .selection_decision
            .value
            .replace("-", " ")
            .upper()
        ),
    )

    print(
        "Candidate eligible for selection:",
        (
            "YES"
            if (
                comparison.selection_decision
                == ExperimentSelectionDecision
                .ELIGIBLE_FOR_SELECTION
            )
            else "NO"
        ),
    )

    print()
    print(
        "NOTE: Relative improvement does not "
        "make a candidate acceptable. A candidate "
        "must meet the 95% Top-1 quality threshold "
        "and satisfy the validation requirements "
        "before it can be eligible for selection."
    )

    print()
    print(
        "NOTE: This experiment uses a "
        "preliminary development dataset. "
        "It cannot support a validated "
        "overall accuracy claim."
    )

    generated_at = datetime.now(
        timezone.utc
    )

    generated_at_utc = (
        generated_at.isoformat()
    )

    timestamp_for_filename = (
        generated_at.strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    policy_ids = tuple(
        policy_id
        for policy_id, _ in POLICIES
    )

    experiment_record = (
            build_experiment_record(
                comparison=comparison,
                policy_ids=policy_ids,
                chunk_count=len(retrieval_text_index),
                dataset_path=DATASET_PATH,
                dataset_sha256=(
                    dataset_sha256
                ),
                corpus_sha256=(
                    corpus_sha256
                ),
                repository_commit=(
                    repository_commit
                ),
                generated_at_utc=(
                    generated_at_utc
                ),
                embedding_model=(
                    DEFAULT_EMBEDDING_MODEL
                ),
                semantic_weight=(
                    SEMANTIC_WEIGHT
                ),
                lexical_weight=(
                    LEXICAL_WEIGHT
                ),
                baseline_strategy=(
                    "hybrid-semantic-lexical-"
                    "cross-encoder"
                ),
                baseline_embedding_text_strategy=(
                    BODY_ONLY_EMBEDDING
                ),
                baseline_semantic_weight=(
                    SEMANTIC_WEIGHT
                ),
                baseline_lexical_weight=(
                    LEXICAL_WEIGHT
                ),
                baseline_reranker_model=(
                    DEFAULT_RERANKER_MODEL
                ),
                baseline_rerank_depth=(
                    BASELINE_RERANK_DEPTH
                ),
                candidate_strategy=(
                    "hybrid-semantic-lexical-"
                    "cross-encoder"
                ),
                candidate_embedding_text_strategy=(
                    BODY_ONLY_EMBEDDING
                ),
                reranker_model=(
                    DEFAULT_RERANKER_MODEL
                ),
                rerank_depth=(
                    CANDIDATE_RERANK_DEPTH
                ),
            )
        )  

    output_path = (
        f"{EXPERIMENT_OUTPUT_DIRECTORY}/"
        "body-only-rerank-depth-11-v5-"
        "dataset-v1-2-"
        f"{timestamp_for_filename}.json"
    )

    write_experiment_record(
        experiment_record,
        output_path,
    )

    print()
    print("=== EXPERIMENT EVIDENCE ===")
    print(
        "Repository commit:",
        repository_commit,
    )
    print(
        "Dataset SHA-256:",
        dataset_sha256,
    )
    print(
        "Corpus SHA-256:",
        corpus_sha256,
    )
    print(
        "Embedding model:",
        DEFAULT_EMBEDDING_MODEL,
    )
    print(
        "Reranker model:",
        DEFAULT_RERANKER_MODEL,
    )
    print(
        "Evaluation Top-K:",
        evaluation_config.top_k,
    )
    print(
        "Experiment record:",
        output_path,
    )
    print(
        "Baseline embedding text strategy:",
        BODY_ONLY_EMBEDDING,
    )
    print(
        "Candidate embedding text strategy:",
        BODY_ONLY_EMBEDDING,
    )
    print(
        "Baseline rerank depth:",
        BASELINE_RERANK_DEPTH,
    )
    print(
        "Candidate rerank depth:",
        CANDIDATE_RERANK_DEPTH,
    )


if __name__ == "__main__":
    main()