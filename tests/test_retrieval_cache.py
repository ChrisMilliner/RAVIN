from dataclasses import replace
from pathlib import Path
import pytest
from backend.ingestion.models import (
    PolicyChunk,
)
from backend.retrieval.cache import (
    RetrievalCacheInvalidError,
    RetrievalCacheStaleError,
    load_retrieval_cache,
    save_retrieval_cache,
)
from backend.retrieval.models import (
    IndexedPolicyChunk,
)
from datetime import (
    datetime,
    timedelta,
    timezone,
)

def _indexed_chunk(
    policy_id: str = "208",
) -> IndexedPolicyChunk:
    chunk = PolicyChunk(
        policy_id=policy_id,
        policy_title=(
            "Example Policy"
        ),
        source_url=(
            "https://example.test/"
            f"{policy_id}"
        ),
        status="Current",
        effective_date=(
            "1 January 2026"
        ),
        review_date=None,
        chunk_index=0,
        text="Approved policy evidence.",
        heading_path=(
            "Section 1",
        ),
    )

    return IndexedPolicyChunk(
        chunk=chunk,
        retrieval_text=(
            "Example Policy\n"
            "Section 1\n"
            "Approved policy evidence."
        ),
        embedding=(
            0.1,
            0.2,
            0.3,
        ),
    )

def test_cache_older_than_24_hours_is_rejected(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "cache.json.gz"
    )

    metadata = save_retrieval_cache(
        path,
        (
            _indexed_chunk(),
        ),
        embedding_provider=(
            "sentence_transformer"
        ),
        embedding_model="model-a",
    )

    created_at = datetime.fromisoformat(
        metadata.created_at_utc
    )

    with pytest.raises(
        RetrievalCacheStaleError,
        match="stale",
    ):
        load_retrieval_cache(
            path,
            embedding_provider=(
                "sentence_transformer"
            ),
            embedding_model="model-a",
            current_time=(
                created_at
                + timedelta(
                    hours=24,
                    seconds=1,
                )
            ),
        )


def test_cache_younger_than_24_hours_is_accepted(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "cache.json.gz"
    )

    metadata = save_retrieval_cache(
        path,
        (
            _indexed_chunk(),
        ),
        embedding_provider=(
            "sentence_transformer"
        ),
        embedding_model="model-a",
    )

    created_at = datetime.fromisoformat(
        metadata.created_at_utc
    )

    loaded = load_retrieval_cache(
        path,
        embedding_provider=(
            "sentence_transformer"
        ),
        embedding_model="model-a",
        current_time=(
            created_at
            + timedelta(
                hours=23,
                minutes=59,
            )
        ),
    )

    assert loaded is not None

def test_cache_round_trip(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "cache.json.gz"
    )

    indexed_chunks = (
        _indexed_chunk(),
        _indexed_chunk(
            "220"
        ),
    )

    saved_metadata = (
        save_retrieval_cache(
            path,
            indexed_chunks,
            embedding_provider=(
                "sentence_transformer"
            ),
            embedding_model="model-a",
            restricted_policy_ids=(
                "268",
            ),
        )
    )

    loaded = load_retrieval_cache(
        path,
        embedding_provider=(
            "sentence_transformer"
        ),
        embedding_model="model-a",
    )

    assert loaded is not None

    assert (
        loaded.indexed_chunks
        == indexed_chunks
    )

    assert (
        loaded.metadata.policy_count
        == 2
    )

    assert (
        loaded.metadata.chunk_count
        == 2
    )

    assert (
        loaded.metadata.restricted_policy_ids
        == ("268",)
    )

    assert (
        loaded.metadata
        == saved_metadata
    )

def test_missing_cache_returns_none(
    tmp_path: Path,
):
    result = load_retrieval_cache(
        tmp_path
        / "missing.json.gz",
        embedding_provider=(
            "sentence_transformer"
        ),
        embedding_model="model-a",
    )

    assert result is None

def test_incompatible_embedding_model_returns_none(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "cache.json.gz"
    )

    save_retrieval_cache(
        path,
        (
            _indexed_chunk(),
        ),
        embedding_provider=(
            "sentence_transformer"
        ),
        embedding_model="model-a",
    )

    result = load_retrieval_cache(
        path,
        embedding_provider=(
            "sentence_transformer"
        ),
        embedding_model="model-b",
    )

    assert result is None

def test_corrupt_cache_is_rejected(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "cache.json.gz"
    )

    save_retrieval_cache(
        path,
        (
            _indexed_chunk(),
        ),
        embedding_provider=(
            "sentence_transformer"
        ),
        embedding_model="model-a",
    )

    path.write_bytes(
        b"corrupt"
    )

    with pytest.raises(
        RetrievalCacheInvalidError,
        match="checksum",
    ):
        load_retrieval_cache(
            path,
            embedding_provider=(
                "sentence_transformer"
            ),
            embedding_model="model-a",
        )

def test_non_current_chunk_is_rejected(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "cache.json.gz"
    )

    current = _indexed_chunk()

    non_current = replace(
        current,
        chunk=replace(
            current.chunk,
            status="Superseded",
        ),
    )

    save_retrieval_cache(
        path,
        (
            non_current,
        ),
        embedding_provider=(
            "sentence_transformer"
        ),
        embedding_model="model-a",
    )

    with pytest.raises(
        RetrievalCacheInvalidError,
        match="non-current",
    ):
        load_retrieval_cache(
            path,
            embedding_provider=(
                "sentence_transformer"
            ),
            embedding_model="model-a",
        )