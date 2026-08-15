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
from backend.retrieval.index import (
    build_semantic_index,
    search_semantic_index,
)
from backend.retrieval.sentence_transformer_provider import (
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
DATASET_PATH = (
    "evaluation/retrieval_baseline.json"
)
SEMANTIC_WEIGHT = 0.85
LEXICAL_WEIGHT = 0.15

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
    print(
        "=== RAVIN RETRIEVAL EXPERIMENT ==="
    )
    print()

    questions = load_evaluation_questions(
        DATASET_PATH
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
        "=== BUILDING SHARED SEMANTIC INDEX ==="
    )

    embedding_provider = (
        SentenceTransformerEmbeddingProvider()
    )

    semantic_index = build_semantic_index(
        tuple(all_chunks),
        embedding_provider,
    )

    print(
        "Indexed chunks:",
        len(semantic_index),
    )

    evaluation_config = EvaluationConfig()

    def retrieve_baseline(
        question: str,
        top_k: int,
    ):
        return search_semantic_index(
            semantic_index,
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
            semantic_index,
            query=question,
            embedding_provider=(
                embedding_provider
            ),
            top_k=top_k,
            semantic_weight=SEMANTIC_WEIGHT,
            lexical_weight=LEXICAL_WEIGHT,
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

    candidate = run_retrieval_evaluation(
        questions,
        retrieve_candidate,
        evaluation_config,
    )

    print(
        "Candidate evaluation complete."
    )

    experiment_config = (
        RetrievalExperimentConfig(
            experiment_name=(
                "Hybrid semantic lexical v1"
            ),
            baseline_name=(
                "MiniLM semantic baseline"
            ),
            candidate_name=(
                "MiniLM + lexical token "
                "coverage 85/15 v1"
            ),
            dataset_name=(
                "RAVIN Preliminary Retrieval "
                "Development Baseline v1"
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
            baseline,
            candidate,
            experiment_config,
        )
    )

    print_evaluation_metrics(
        "BASELINE - MINILM SEMANTIC",
        baseline,
    )

    print_evaluation_metrics(
        "CANDIDATE - HYBRID 85/15",
        candidate,
    )

    print()
    print("=== METRIC COMPARISON ===")
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


if __name__ == "__main__":
    main()