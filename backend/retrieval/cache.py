"""
Persist and restore RAVIN's production retrieval corpus.

The cache stores traceable policy chunks together with their precomputed
embedding vectors. A successful cache load allows normal application
startup to avoid repeating live policy acquisition, chunking, and
document embedding.

Cache files are local runtime artefacts rather than repository source.
A SHA-256 sidecar detects accidental corruption before cached data is
accepted.
"""

import gzip
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path
from backend.ingestion.models import (
    PolicyChunk,
)
from backend.retrieval.models import (
    IndexedPolicyChunk,
)

CACHE_SCHEMA_VERSION = 1
DEFAULT_CACHE_MAX_AGE = timedelta(
    hours=24
)

@dataclass(frozen=True)
class RetrievalCacheMetadata:
    """Describe one validated retrieval-cache snapshot."""

    created_at_utc: str
    embedding_provider: str
    embedding_model: str
    policy_count: int
    chunk_count: int
    restricted_policy_ids: tuple[str, ...]

@dataclass(frozen=True)
class CachedRetrievalCorpus:
    """Contain a validated cached production retrieval corpus."""

    metadata: RetrievalCacheMetadata
    indexed_chunks: tuple[
        IndexedPolicyChunk,
        ...
    ]

class RetrievalCacheInvalidError(
    RuntimeError
):
    """Signal that an existing retrieval cache cannot be trusted."""

    pass

class RetrievalCacheStaleError(
    RuntimeError
):
    """Signal that a valid retrieval cache is too old for production use."""

    pass

def save_retrieval_cache(
    path: Path,
    indexed_chunks: tuple[
        IndexedPolicyChunk,
        ...
    ],
    *,
    embedding_provider: str,
    embedding_model: str,
    restricted_policy_ids: tuple[
        str,
        ...
    ] = (),
) -> RetrievalCacheMetadata:
    """Persist one complete indexed retrieval corpus atomically."""
    if not indexed_chunks:
        raise ValueError(
            "Retrieval cache cannot store an empty index."
        )

    if not embedding_provider.strip():
        raise ValueError(
            "Cache embedding provider cannot be empty."
        )

    if not embedding_model.strip():
        raise ValueError(
            "Cache embedding model cannot be empty."
        )

    policy_ids = {
        indexed_chunk.chunk.policy_id
        for indexed_chunk in indexed_chunks
    }

    metadata = RetrievalCacheMetadata(
        created_at_utc=(
            datetime.now(
                timezone.utc
            )
            .replace(
                microsecond=0
            )
            .isoformat()
        ),
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        policy_count=len(
            policy_ids
        ),
        chunk_count=len(
            indexed_chunks
        ),
        restricted_policy_ids=tuple(
            restricted_policy_ids
        ),
    )

    payload = {
        "schema_version": (
            CACHE_SCHEMA_VERSION
        ),
        "metadata": {
            "created_at_utc": (
                metadata.created_at_utc
            ),
            "embedding_provider": (
                metadata.embedding_provider
            ),
            "embedding_model": (
                metadata.embedding_model
            ),
            "policy_count": (
                metadata.policy_count
            ),
            "chunk_count": (
                metadata.chunk_count
            ),
            "restricted_policy_ids": list(
                metadata.restricted_policy_ids
            ),
        },
        "indexed_chunks": [
            _serialize_indexed_chunk(
                indexed_chunk
            )
            for indexed_chunk
            in indexed_chunks
        ],
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    checksum_path = _checksum_path(
        path
    )

    temporary_checksum_path = (
        checksum_path.with_suffix(
            checksum_path.suffix + ".tmp"
        )
    )

    try:
        with gzip.open(
            temporary_path,
            "wt",
            encoding="utf-8",
        ) as cache_file:
            json.dump(
                payload,
                cache_file,
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            )

        checksum = _file_sha256(
            temporary_path
        )

        temporary_checksum_path.write_text(
            checksum + "\n",
            encoding="utf-8",
        )

        os.replace(
            temporary_path,
            path,
        )

        os.replace(
            temporary_checksum_path,
            checksum_path,
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

        if (
            temporary_checksum_path
            .exists()
        ):
            temporary_checksum_path.unlink()

    return metadata

def load_retrieval_cache(
    path: Path,
    *,
    embedding_provider: str,
    embedding_model: str,
    max_age: timedelta = (
        DEFAULT_CACHE_MAX_AGE
    ),
    current_time: (
        datetime | None
    ) = None,
) -> CachedRetrievalCorpus | None:
    """Load and validate a compatible retrieval cache when one exists."""
    if not path.exists():
        return None

    checksum_path = _checksum_path(
        path
    )

    if not checksum_path.exists():
        raise RetrievalCacheInvalidError(
            "Retrieval cache checksum is missing."
        )

    expected_checksum = (
        checksum_path
        .read_text(
            encoding="utf-8"
        )
        .strip()
    )

    actual_checksum = _file_sha256(
        path
    )

    if (
        not expected_checksum
        or expected_checksum
        != actual_checksum
    ):
        raise RetrievalCacheInvalidError(
            "Retrieval cache checksum validation failed."
        )

    try:
        with gzip.open(
            path,
            "rt",
            encoding="utf-8",
        ) as cache_file:
            payload = json.load(
                cache_file
            )

    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        raise RetrievalCacheInvalidError(
            "Retrieval cache could not be decoded."
        ) from error

    if (
        payload.get(
            "schema_version"
        )
        != CACHE_SCHEMA_VERSION
    ):
        raise RetrievalCacheInvalidError(
            "Retrieval cache schema version is incompatible."
        )

    metadata_payload = payload.get(
        "metadata"
    )

    if not isinstance(
        metadata_payload,
        dict,
    ):
        raise RetrievalCacheInvalidError(
            "Retrieval cache metadata is missing."
        )

    metadata = RetrievalCacheMetadata(
        created_at_utc=str(
            metadata_payload.get(
                "created_at_utc",
                "",
            )
        ),
        embedding_provider=str(
            metadata_payload.get(
                "embedding_provider",
                "",
            )
        ),
        embedding_model=str(
            metadata_payload.get(
                "embedding_model",
                "",
            )
        ),
        policy_count=int(
            metadata_payload.get(
                "policy_count",
                0,
            )
        ),
        chunk_count=int(
            metadata_payload.get(
                "chunk_count",
                0,
            )
        ),
        restricted_policy_ids=tuple(
            str(policy_id)
            for policy_id
            in metadata_payload.get(
                "restricted_policy_ids",
                [],
            )
        ),
    )

    if (
        metadata.embedding_provider
        != embedding_provider
        or metadata.embedding_model
        != embedding_model
    ):
        return None

    if max_age.total_seconds() <= 0:
        raise ValueError(
            "Retrieval cache maximum age must be greater than zero."
        )

    try:
        created_at = (
            datetime.fromisoformat(
                metadata.created_at_utc
            )
        )
    except ValueError as error:
        raise RetrievalCacheInvalidError(
            "Retrieval cache creation timestamp is invalid."
        ) from error

    if created_at.tzinfo is None:
        raise RetrievalCacheInvalidError(
            "Retrieval cache creation timestamp must include a timezone."
        )

    resolved_current_time = (
        datetime.now(
            timezone.utc
        )
        if current_time is None
        else current_time
    )

    if (
        resolved_current_time.tzinfo
        is None
    ):
        raise ValueError(
            "Current cache-validation time must include a timezone."
        )

    cache_age = (
        resolved_current_time
        - created_at
    )

    if cache_age > max_age:
        raise RetrievalCacheStaleError(
            "Retrieval cache is stale and must be refreshed."
        )

    indexed_payload = payload.get(
        "indexed_chunks"
    )

    if not isinstance(
        indexed_payload,
        list,
    ):
        raise RetrievalCacheInvalidError(
            "Retrieval cache index is missing."
        )

    try:
        indexed_chunks = tuple(
            _deserialize_indexed_chunk(
                item
            )
            for item in indexed_payload
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise RetrievalCacheInvalidError(
            "Retrieval cache index is invalid."
        ) from error

    _validate_loaded_corpus(
        metadata,
        indexed_chunks,
    )

    return CachedRetrievalCorpus(
        metadata=metadata,
        indexed_chunks=indexed_chunks,
    )

def _serialize_indexed_chunk(
    indexed_chunk: IndexedPolicyChunk,
) -> dict:
    chunk = indexed_chunk.chunk

    return {
        "chunk": {
            "policy_id": (
                chunk.policy_id
            ),
            "policy_title": (
                chunk.policy_title
            ),
            "source_url": (
                chunk.source_url
            ),
            "status": (
                chunk.status
            ),
            "effective_date": (
                chunk.effective_date
            ),
            "review_date": (
                chunk.review_date
            ),
            "chunk_index": (
                chunk.chunk_index
            ),
            "text": (
                chunk.text
            ),
            "heading_path": list(
                chunk.heading_path
            ),
        },
        "retrieval_text": (
            indexed_chunk.retrieval_text
        ),
        "embedding": list(
            indexed_chunk.embedding
        ),
    }

def _deserialize_indexed_chunk(
    payload: dict,
) -> IndexedPolicyChunk:
    chunk_payload = payload[
        "chunk"
    ]

    chunk = PolicyChunk(
        policy_id=str(
            chunk_payload[
                "policy_id"
            ]
        ),
        policy_title=str(
            chunk_payload[
                "policy_title"
            ]
        ),
        source_url=str(
            chunk_payload[
                "source_url"
            ]
        ),
        status=str(
            chunk_payload[
                "status"
            ]
        ),
        effective_date=(
            None
            if (
                chunk_payload[
                    "effective_date"
                ]
                is None
            )
            else str(
                chunk_payload[
                    "effective_date"
                ]
            )
        ),
        review_date=(
            None
            if (
                chunk_payload[
                    "review_date"
                ]
                is None
            )
            else str(
                chunk_payload[
                    "review_date"
                ]
            )
        ),
        chunk_index=int(
            chunk_payload[
                "chunk_index"
            ]
        ),
        text=str(
            chunk_payload[
                "text"
            ]
        ),
        heading_path=tuple(
            str(heading)
            for heading
            in chunk_payload[
                "heading_path"
            ]
        ),
    )

    embedding = tuple(
        float(value)
        for value in payload[
            "embedding"
        ]
    )

    if not embedding:
        raise ValueError(
            "Cached embedding cannot be empty."
        )

    return IndexedPolicyChunk(
        chunk=chunk,
        retrieval_text=str(
            payload[
                "retrieval_text"
            ]
        ),
        embedding=embedding,
    )

def _validate_loaded_corpus(
    metadata: RetrievalCacheMetadata,
    indexed_chunks: tuple[
        IndexedPolicyChunk,
        ...
    ],
) -> None:
    if not indexed_chunks:
        raise RetrievalCacheInvalidError(
            "Retrieval cache contains no indexed chunks."
        )

    if (
        metadata.chunk_count
        != len(
            indexed_chunks
        )
    ):
        raise RetrievalCacheInvalidError(
            "Retrieval cache chunk count does not match metadata."
        )

    policy_ids = {
        indexed_chunk.chunk.policy_id
        for indexed_chunk
        in indexed_chunks
    }

    if (
        metadata.policy_count
        != len(
            policy_ids
        )
    ):
        raise RetrievalCacheInvalidError(
            "Retrieval cache policy count does not match metadata."
        )

    if any(
        indexed_chunk.chunk.status
        .casefold()
        != "current"
        for indexed_chunk
        in indexed_chunks
    ):
        raise RetrievalCacheInvalidError(
            "Retrieval cache contains a non-current policy chunk."
        )

    if any(
        not indexed_chunk.embedding
        for indexed_chunk
        in indexed_chunks
    ):
        raise RetrievalCacheInvalidError(
            "Retrieval cache contains an empty embedding."
        )

def _checksum_path(
    path: Path,
) -> Path:
    return Path(
        str(path) + ".sha256"
    )

def _file_sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as source:
        while True:
            block = source.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()