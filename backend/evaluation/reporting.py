import hashlib
import json
import subprocess
from pathlib import Path
from backend.evaluation.experiment_models import (
    ExperimentSelectionDecision,
    RetrievalExperimentComparison,
)
from backend.retrieval.models import (
    IndexedPolicyChunk,
)

def calculate_file_sha256(
    file_path: str | Path,
) -> str:
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
    embedding_model: str,
    semantic_weight: float,
    lexical_weight: float,
) -> dict[str, object]:
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

    if not embedding_model.strip():
        raise ValueError(
            "Embedding model cannot be empty."
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

    config = comparison.config

    candidate_eligible = (
        comparison.selection_decision
        == ExperimentSelectionDecision
        .ELIGIBLE_FOR_SELECTION
    )

    return {
        "schema_version": 1,
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
            "embedding_model": embedding_model,
            "baseline": {
                "strategy": "semantic",
                "semantic_weight": 1.0,
                "lexical_weight": 0.0,
            },
            "candidate": {
                "strategy": (
                    "hybrid-semantic-lexical"
                ),
                "semantic_weight": (
                    semantic_weight
                ),
                "lexical_weight": (
                    lexical_weight
                ),
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
    record: dict[str, object],
    output_path: str | Path,
) -> None:
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