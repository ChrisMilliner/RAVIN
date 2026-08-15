import hashlib
import json
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

def build_experiment_record(
    comparison: RetrievalExperimentComparison,
    policy_ids: tuple[str, ...],
    chunk_count: int,
    dataset_path: str,
    dataset_sha256: str,
    corpus_sha256: str,
    repository_commit: str,
    generated_at_utc: str,
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