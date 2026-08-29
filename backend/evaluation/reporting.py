"""
Create reproducible records for RAVIN retrieval experiments.

This module captures dataset and corpus hashes, repository commit
identity, evaluation configuration, and experiment results so an
optimisation decision can be traced back to the exact evaluated state.

Experiment recording requires a clean Git working tree to reduce the
risk of producing results that cannot later be reproduced.
"""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any
from backend.evaluation.experiment_models import (
    DatasetValidationStatus,
    ExperimentSelectionDecision,
    RetrievalExperimentComparison,
)
from backend.retrieval.models import (
    IndexedPolicyChunk,
)
from backend.evaluation.models import (
    GroundedOverviewEvaluationConfig,
    GroundedOverviewEvaluationResult,
)

def calculate_file_sha256(
    file_path: str | Path,
) -> str:
    """Calculate the SHA-256 fingerprint of a file used by evaluation.
    """
    path = Path(file_path)

    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()

def calculate_corpus_sha256(
    indexed_chunks: tuple[
        IndexedPolicyChunk,
        ...
    ],
) -> str:
    """Calculate a deterministic fingerprint of the indexed retrieval corpus.
    """
    if not indexed_chunks:
        raise ValueError(
            "Cannot fingerprint an empty corpus."
        )

    hasher = hashlib.sha256()

    for indexed_chunk in indexed_chunks:
        chunk = indexed_chunk.chunk

        hasher.update(
            chunk.policy_id.encode("utf-8")
        )
        hasher.update(b"\n")

        hasher.update(
            str(chunk.chunk_index).encode(
                "utf-8"
            )
        )
        hasher.update(b"\n")

        hasher.update(
            indexed_chunk.retrieval_text.encode(
                "utf-8"
            )
        )

        hasher.update(
            b"\n---RAVIN-CHUNK---\n"
        )

    return hasher.hexdigest()

def ensure_clean_git_working_tree() -> None:
    """Require a clean Git working tree before recording reproducible experiments.
    """
    result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    if result.stdout.strip():
        raise RuntimeError(
            "Experiment records require a clean "
            "Git working tree."
        )

def get_repository_commit() -> str:
    """Return the Git commit identifying the evaluated repository state.
    """
    result = subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    commit = result.stdout.strip()

    if not commit:
        raise RuntimeError(
            "Could not determine repository commit."
        )

    return commit

def build_experiment_record(
    comparison: RetrievalExperimentComparison,
    policy_ids: tuple[str, ...],
    chunk_count: int,
    dataset_path: str,
    dataset_sha256: str,
    corpus_sha256: str,
    repository_commit: str,
    generated_at_utc: str,
    embedding_provider: str,
    embedding_model: str,
    semantic_weight: float,
    lexical_weight: float,
    baseline_strategy: str = "semantic",
    baseline_semantic_weight: float = 1.0,
    baseline_lexical_weight: float = 0.0,
    candidate_strategy: str = (
        "hybrid-semantic-lexical"
    ),
    baseline_embedding_text_strategy: str = (
        "retrieval-text"
    ),
    candidate_embedding_text_strategy: str = (
        "retrieval-text"
    ),
    baseline_reranker_provider: str | None = None,
    baseline_reranker_model: str | None = None,
    baseline_rerank_depth: int | None = None,
    reranker_provider: str | None = None,
    reranker_model: str | None = None,
    rerank_depth: int | None = None,
    grounded_overview_config: (
        GroundedOverviewEvaluationConfig | None
    ) = None,
    grounded_overview_evaluation: (
        GroundedOverviewEvaluationResult | None
    ) = None,
) -> dict[str, Any]:
    """Build the structured reproducibility record for a retrieval experiment.

    The record combines dataset and corpus fingerprints, repository
    provenance, provider configuration, metrics, gates, and selection
    decisions.
    """
    if not policy_ids:
        raise ValueError(
            "Experiment record requires policy IDs."
        )

    if chunk_count <= 0:
        raise ValueError(
            "Experiment record chunk count must "
            "be greater than zero."
        )

    if not dataset_path.strip():
        raise ValueError(
            "Dataset path cannot be empty."
        )

    if not dataset_sha256.strip():
        raise ValueError(
            "Dataset fingerprint cannot be empty."
        )

    if not corpus_sha256.strip():
        raise ValueError(
            "Corpus fingerprint cannot be empty."
        )

    if not repository_commit.strip():
        raise ValueError(
            "Repository commit cannot be empty."
        )

    if not generated_at_utc.strip():
        raise ValueError(
            "Experiment timestamp cannot be empty."
        )

    if not embedding_provider.strip():
        raise ValueError(
            "Embedding provider cannot be empty."
        )

    if not embedding_model.strip():
        raise ValueError(
            "Embedding model cannot be empty."
        )

    if not baseline_strategy.strip():
        raise ValueError(
            "Baseline strategy cannot be empty."
        )

    if not candidate_strategy.strip():
        raise ValueError(
            "Candidate strategy cannot be empty."
        )

    if not baseline_embedding_text_strategy.strip():
        raise ValueError(
            "Baseline embedding text strategy "
            "cannot be empty."
        )

    if not candidate_embedding_text_strategy.strip():
        raise ValueError(
            "Candidate embedding text strategy "
            "cannot be empty."
        )

    if not 0.0 <= baseline_semantic_weight <= 1.0:
        raise ValueError(
            "Baseline semantic weight must be "
            "between 0 and 1."
        )

    if not 0.0 <= baseline_lexical_weight <= 1.0:
        raise ValueError(
            "Baseline lexical weight must be "
            "between 0 and 1."
        )

    if abs(
        baseline_semantic_weight
        + baseline_lexical_weight
        - 1.0
    ) > 1e-9:
        raise ValueError(
            "Baseline retrieval weights must "
            "sum to 1."
        )

    if (
        baseline_reranker_model is None
        and baseline_rerank_depth is not None
    ):
        raise ValueError(
            "Baseline rerank depth requires "
            "a baseline reranker model."
        )

    if (
        baseline_reranker_model is not None
        and not baseline_reranker_model.strip()
    ):
        raise ValueError(
            "Baseline reranker model cannot be empty."
        )

    if (
        baseline_reranker_model is not None
        and baseline_rerank_depth is None
    ):
        raise ValueError(
            "Baseline reranker model requires "
            "a baseline rerank depth."
        )

    if (
        baseline_rerank_depth is not None
        and baseline_rerank_depth <= 0
    ):
        raise ValueError(
            "Baseline rerank depth must be "
            "greater than zero."
        )

    if (
        reranker_model is None
        and rerank_depth is not None
    ):
        raise ValueError(
            "Rerank depth requires a reranker model."
        )

    if (
        reranker_model is not None
        and not reranker_model.strip()
    ):
        raise ValueError(
            "Reranker model cannot be empty."
        )

    if (
        reranker_model is not None
        and rerank_depth is None
    ):
        raise ValueError(
            "Reranker model requires a rerank depth."
        )

    if (
        rerank_depth is not None
        and rerank_depth <= 0
    ):
        raise ValueError(
            "Rerank depth must be greater than zero."
        )

    if (
    reranker_model is not None
    and reranker_provider is None
    ):
        raise ValueError(
            "Reranker model requires "
            "a reranker provider."
        )

    if (
        reranker_provider is not None
        and not reranker_provider.strip()
    ):
        raise ValueError(
            "Reranker provider cannot be empty."
        )

    if (
        reranker_provider is not None
        and reranker_model is None
    ):
        raise ValueError(
            "Reranker provider requires "
            "a reranker model."
        )

    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError(
            "Semantic weight must be between 0 and 1."
        )

    if not 0.0 <= lexical_weight <= 1.0:
        raise ValueError(
            "Lexical weight must be between 0 and 1."
        )

    if abs(
        semantic_weight
        + lexical_weight
        - 1.0
    ) > 1e-9:
        raise ValueError(
            "Retrieval weights must sum to 1."
        )

    if (
        (grounded_overview_config is None)
        != (grounded_overview_evaluation is None)
    ):
        raise ValueError(
            "Grounded overview configuration and "
            "evaluation must be provided together."
        )

    if (
        grounded_overview_config is not None
        and grounded_overview_evaluation is not None
    ):
        if abs(
            grounded_overview_config.pass_threshold
            - grounded_overview_evaluation.pass_threshold
        ) > 1e-12:
            raise ValueError(
                "Grounded overview configuration and "
                "evaluation thresholds must match."
            )

        if (
            grounded_overview_evaluation.total_questions
            != comparison.population
            .grounded_overview_questions
        ):
            raise ValueError(
                "Grounded overview evaluation question "
                "count must match the evaluation population."
            )

    config = comparison.config

    candidate_eligible = (
        comparison.selection_decision
        == ExperimentSelectionDecision
        .ELIGIBLE_FOR_SELECTION
    )

    grounded_overview_metrics = None

    if (
        grounded_overview_config is not None
        and grounded_overview_evaluation is not None
    ):
        grounded_overview_metrics = {
            "top_k": (
                grounded_overview_config.top_k
            ),
            "quality_threshold": (
                grounded_overview_evaluation
                .pass_threshold
            ),
            "total_questions": (
                grounded_overview_evaluation
                .total_questions
            ),
            "passed_questions": (
                grounded_overview_evaluation
                .passed_questions
            ),
            "question_pass_rate": (
                grounded_overview_evaluation
                .question_pass_rate
            ),
            "total_evidence_groups": (
                grounded_overview_evaluation
                .total_groups
            ),
            "covered_evidence_groups": (
                grounded_overview_evaluation
                .covered_groups
            ),
            "evidence_group_coverage": (
                grounded_overview_evaluation
                .evidence_group_coverage
            ),
            "quality_gate_passed": (
                grounded_overview_evaluation.passed
            ),
            "validated_dataset_gate_passed": (
                grounded_overview_evaluation.passed
                and comparison.config.dataset_status
                == DatasetValidationStatus.HUMAN_VALIDATED
            ),
            "per_question": [
                {
                    "question_id": (
                        question_result.question_id
                    ),
                    "total_groups": (
                        question_result.total_groups
                    ),
                    "covered_groups": (
                        question_result.covered_groups
                    ),
                    "evidence_coverage": (
                        question_result
                        .evidence_coverage
                    ),
                    "passed": (
                        question_result.passed
                    ),
                    "groups": [
                        {
                            "group_id": (
                                group_result.group_id
                            ),
                            "covered": (
                                group_result.covered
                            ),
                        }
                        for group_result in (
                            question_result.group_results
                        )
                    ],
                }
                for question_result in (
                    grounded_overview_evaluation
                    .question_results
                )
            ],
        }

    return {
        "schema_version": 3,
        "experiment": {
            "name": config.experiment_name,
            "baseline": config.baseline_name,
            "candidate": config.candidate_name,
            "top_k": config.top_k,
            "quality_threshold": (
                config.quality_threshold
            ),
        },
        "retrieval_configuration": {
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "baseline": {
                "strategy": baseline_strategy,
                "reranker_provider": (
                    baseline_reranker_provider
                ),
                "embedding_text_strategy": (
                    baseline_embedding_text_strategy
                ),
                "semantic_weight": (
                    baseline_semantic_weight
                ),
                "lexical_weight": (
                    baseline_lexical_weight
                ),
                "reranker_model": (
                    baseline_reranker_model
                ),
                "rerank_depth": (
                    baseline_rerank_depth
                ),
            },
            "candidate": {
                "strategy": candidate_strategy,
                "embedding_text_strategy": (
                    candidate_embedding_text_strategy
                ),
                "semantic_weight": (
                    semantic_weight
                ),
                "lexical_weight": (
                    lexical_weight
                ),
                "reranker_provider": reranker_provider,
                "reranker_model": reranker_model,
                "rerank_depth": rerank_depth,
            },
        },
        "dataset": {
            "name": config.dataset_name,
            "status": (
                config.dataset_status.value
            ),
            "path": dataset_path,
            "sha256": dataset_sha256,
        },
        "evaluation_population": {
            "dataset_questions": (
                comparison.population.dataset_questions
            ),
            "direct_answer_questions": (
                comparison.population
                .direct_answer_questions
            ),
            "grounded_overview_questions": (
                comparison.population
                .grounded_overview_questions
            ),
            "clarify_questions": (
                comparison.population
                .clarify_questions
            ),
            "no_grounded_answer_questions": (
                comparison.population
                .no_grounded_answer_questions
            ),
            "ranking_metric_scope": (
                "direct_answer"
            ),
        },
        "grounded_overview_metrics": (
            grounded_overview_metrics
        ),
        "corpus": {
            "policy_ids": list(policy_ids),
            "chunk_count": chunk_count,
            "sha256": corpus_sha256,
        },
        "provenance": {
            "repository_commit": (
                repository_commit
            ),
            "generated_at_utc": (
                generated_at_utc
            ),
        },
        "baseline_metrics": {
            "top_1_accuracy": (
                comparison.top_1.baseline
            ),
            "hit_at_k": (
                comparison.hit_at_k.baseline
            ),
            "mrr": comparison.mrr.baseline,
        },
        "candidate_metrics": {
            "top_1_accuracy": (
                comparison.top_1.candidate
            ),
            "hit_at_k": (
                comparison.hit_at_k.candidate
            ),
            "mrr": comparison.mrr.candidate,
        },
        "metric_deltas": {
            "top_1_accuracy": (
                comparison.top_1.delta
            ),
            "hit_at_k": (
                comparison.hit_at_k.delta
            ),
            "mrr": comparison.mrr.delta,
        },
        "question_rank_changes": [
            {
                "question_id": (
                    change.question_id
                ),
                "baseline_rank": (
                    change.baseline_rank
                ),
                "candidate_rank": (
                    change.candidate_rank
                ),
            }
            for change in (
                comparison.question_rank_changes
            )
        ],
        "quality_decision": {
            "relative_direction": (
                comparison.direction.value
            ),
            "quality_gate_passed": (
                comparison.quality_gate_passed
            ),
            "validated_dataset_gate_passed": (
                comparison
                .validated_dataset_gate_passed
            ),
            "selection_decision": (
                comparison
                .selection_decision
                .value
            ),
            "candidate_eligible_for_selection": (
                candidate_eligible
            ),
        },
    }

def write_experiment_record(
    record: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Write a structured experiment record as formatted JSON.
    """
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            record,
            file,
            indent=2,
        )

        file.write("\n")