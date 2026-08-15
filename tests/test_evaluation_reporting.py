import hashlib
import json
import subprocess
import pytest
from backend.evaluation.experiment_models import (
    DatasetValidationStatus,
    ExperimentDirection,
    ExperimentSelectionDecision,
    MetricComparison,
    QuestionRankChange,
    RetrievalExperimentComparison,
    RetrievalExperimentConfig,
)
from backend.evaluation.reporting import (
    build_experiment_record,
    calculate_corpus_sha256,
    calculate_file_sha256,
    ensure_clean_git_working_tree,
    get_repository_commit,
    write_experiment_record,
)
from backend.ingestion.models import PolicyChunk
from backend.retrieval.models import IndexedPolicyChunk

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)
SEMANTIC_WEIGHT = 0.85
LEXICAL_WEIGHT = 0.15

def make_indexed_chunk(
    policy_id: str = "208",
    chunk_index: int = 0,
    text: str = "Academic dress requirements.",
) -> IndexedPolicyChunk:
    chunk = PolicyChunk(
        policy_id=policy_id,
        policy_title="Academic Dress Policy",
        source_url=(
            "https://policies.latrobe.edu.au/"
            f"document/view.php?id={policy_id}"
        ),
        status="Current",
        effective_date=None,
        review_date=None,
        chunk_index=chunk_index,
        text=text,
        heading_path=(
            "Section 6 - Procedures",
        ),
    )

    return IndexedPolicyChunk(
        chunk=chunk,
        retrieval_text=(
            "Academic Dress Policy\n"
            "Section 6 - Procedures\n"
            f"{text}"
        ),
        embedding=(1.0, 0.0),
    )

def make_comparison(
    selection_decision: ExperimentSelectionDecision = (
        ExperimentSelectionDecision
        .REJECT_BELOW_THRESHOLD
    ),
) -> RetrievalExperimentComparison:
    config = RetrievalExperimentConfig(
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
        top_k=5,
        quality_threshold=0.95,
    )

    return RetrievalExperimentComparison(
        config=config,
        top_1=MetricComparison(
            baseline=0.6667,
            candidate=0.7667,
        ),
        hit_at_k=MetricComparison(
            baseline=0.8667,
            candidate=0.9333,
        ),
        mrr=MetricComparison(
            baseline=0.7361,
            candidate=0.8361,
        ),
        question_rank_changes=(
            QuestionRankChange(
                question_id="RB003",
                baseline_rank=None,
                candidate_rank=2,
            ),
        ),
        direction=ExperimentDirection.IMPROVED,
        selection_decision=(
            selection_decision
        ),
        quality_gate_passed=False,
        validated_dataset_gate_passed=False,
    )

def test_file_sha256_matches_expected_digest(
    tmp_path,
):
    file_path = tmp_path / "dataset.json"
    file_path.write_bytes(
        b'{"questions": []}\n'
    )

    expected = hashlib.sha256(
        b'{"questions": []}\n'
    ).hexdigest()

    actual = calculate_file_sha256(
        file_path
    )

    assert actual == expected

def test_corpus_sha256_is_repeatable():
    indexed_chunks = (
        make_indexed_chunk(),
        make_indexed_chunk(
            policy_id="220",
            chunk_index=1,
            text="Academic progression requirements.",
        ),
    )

    first = calculate_corpus_sha256(
        indexed_chunks
    )

    second = calculate_corpus_sha256(
        indexed_chunks
    )

    assert first == second

def test_corpus_sha256_changes_when_content_changes():
    original = (
        make_indexed_chunk(
            text="Original policy content.",
        ),
    )

    changed = (
        make_indexed_chunk(
            text="Changed policy content.",
        ),
    )

    assert (
        calculate_corpus_sha256(original)
        != calculate_corpus_sha256(changed)
    )

def test_corpus_sha256_rejects_empty_corpus():
    with pytest.raises(
        ValueError,
        match=(
            "Cannot fingerprint an empty corpus."
        ),
    ):
        calculate_corpus_sha256(())

def test_experiment_record_preserves_accuracy_decision():
    comparison = make_comparison()

    record = build_experiment_record(
        comparison=comparison,
        policy_ids=(
            "208",
            "220",
            "76",
            "420",
            "169",
            "340",
        ),
        chunk_count=212,
        dataset_path=(
            "evaluation/retrieval_baseline.json"
        ),
        dataset_sha256="dataset-hash",
        corpus_sha256="corpus-hash",
        repository_commit="abc1234",
        generated_at_utc=(
            "2026-08-15T02:00:00+00:00"
        ),
        embedding_model=EMBEDDING_MODEL,
        semantic_weight=SEMANTIC_WEIGHT,
        lexical_weight=LEXICAL_WEIGHT,
    )

    assert record["schema_version"] == 1

    assert (
        record["candidate_metrics"][
            "top_1_accuracy"
        ]
        == pytest.approx(0.7667)
    )

    assert (
        record["experiment"][
            "quality_threshold"
        ]
        == pytest.approx(0.95)
    )

    assert (
        record["quality_decision"][
            "selection_decision"
        ]
        == "reject-below-threshold"
    )

    assert not (
        record["quality_decision"][
            "candidate_eligible_for_selection"
        ]
    )

def test_experiment_record_marks_eligible_decision():
    comparison = make_comparison(
        ExperimentSelectionDecision
        .ELIGIBLE_FOR_SELECTION
    )

    record = build_experiment_record(
        comparison=comparison,
        policy_ids=("208",),
        chunk_count=1,
        dataset_path="evaluation/test.json",
        dataset_sha256="dataset-hash",
        corpus_sha256="corpus-hash",
        repository_commit="abc1234",
        generated_at_utc=(
            "2026-08-15T02:00:00+00:00"
        ),
        embedding_model=EMBEDDING_MODEL,
        semantic_weight=SEMANTIC_WEIGHT,
        lexical_weight=LEXICAL_WEIGHT,
    )

    assert (
        record["quality_decision"][
            "candidate_eligible_for_selection"
        ]
    )

def test_experiment_record_preserves_provenance():
    comparison = make_comparison()

    record = build_experiment_record(
        comparison=comparison,
        policy_ids=("208", "220"),
        chunk_count=212,
        dataset_path=(
            "evaluation/retrieval_baseline.json"
        ),
        dataset_sha256="dataset-hash",
        corpus_sha256="corpus-hash",
        repository_commit="a1e8d1b",
        generated_at_utc=(
            "2026-08-15T02:00:00+00:00"
        ),
        embedding_model=EMBEDDING_MODEL,
        semantic_weight=SEMANTIC_WEIGHT,
        lexical_weight=LEXICAL_WEIGHT,
    )

    assert record["dataset"]["sha256"] == (
        "dataset-hash"
    )

    assert record["corpus"]["sha256"] == (
        "corpus-hash"
    )

    assert (
        record["provenance"][
            "repository_commit"
        ]
        == "a1e8d1b"
    )

def test_write_experiment_record_creates_json_file(
    tmp_path,
):
    output_path = (
        tmp_path
        / "experiments"
        / "hybrid-v1.json"
    )

    record = {
        "experiment": "hybrid-v1",
        "accuracy": 0.7667,
    }

    write_experiment_record(
        record,
        output_path,
    )

    assert output_path.exists()

    loaded = json.loads(
        output_path.read_text(
            encoding="utf-8",
        )
    )

    assert loaded == record

def test_experiment_record_preserves_retrieval_configuration():
    comparison = make_comparison()

    record = build_experiment_record(
        comparison=comparison,
        policy_ids=("208",),
        chunk_count=1,
        dataset_path=(
            "evaluation/retrieval_baseline.json"
        ),
        dataset_sha256="dataset-hash",
        corpus_sha256="corpus-hash",
        repository_commit="abc1234",
        generated_at_utc=(
            "2026-08-15T02:00:00+00:00"
        ),
        embedding_model=EMBEDDING_MODEL,
        semantic_weight=SEMANTIC_WEIGHT,
        lexical_weight=LEXICAL_WEIGHT,
    )

    configuration = (
        record["retrieval_configuration"]
    )

    assert (
        configuration["embedding_model"]
        == EMBEDDING_MODEL
    )

    assert (
        configuration["baseline"][
            "strategy"
        ]
        == "semantic"
    )

    assert (
        configuration["baseline"][
            "semantic_weight"
        ]
        == pytest.approx(1.0)
    )

    assert (
        configuration["candidate"][
            "strategy"
        ]
        == "hybrid-semantic-lexical"
    )

    assert (
        configuration["candidate"][
            "semantic_weight"
        ]
        == pytest.approx(0.85)
    )

    assert (
        configuration["candidate"][
            "lexical_weight"
        ]
        == pytest.approx(0.15)
    )

def test_clean_git_working_tree_accepts_clean_status(
    monkeypatch,
):
    def fake_run(
        *args,
        **kwargs,
    ):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(
        "backend.evaluation.reporting."
        "subprocess.run",
        fake_run,
    )

    ensure_clean_git_working_tree()

def test_clean_git_working_tree_rejects_dirty_status(
    monkeypatch,
):
    def fake_run(
        *args,
        **kwargs,
    ):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=(
                " M backend/retrieval/hybrid.py\n"
                "?? temporary_file.txt\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(
        "backend.evaluation.reporting."
        "subprocess.run",
        fake_run,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Experiment records require a clean "
            "Git working tree."
        ),
    ):
        ensure_clean_git_working_tree()

def test_repository_commit_returns_current_head(
    monkeypatch,
):
    def fake_run(
        *args,
        **kwargs,
    ):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="b127e21abcdef1234567890\n",
            stderr="",
        )

    monkeypatch.setattr(
        "backend.evaluation.reporting."
        "subprocess.run",
        fake_run,
    )

    commit = get_repository_commit()

    assert commit == (
        "b127e21abcdef1234567890"
    )