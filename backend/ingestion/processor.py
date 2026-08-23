from backend.ingestion.chunking import chunk_policy
from backend.ingestion.config import (
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_CHUNK_SIZE_WORDS,
)
from backend.ingestion.models import (
    IngestionResult,
    IngestionStatus,
    ParsedPolicyContent,
    PolicyContentUnit,
    RawPolicyContent,
)
from backend.ingestion.normalization import normalize_policy_text

def process_policy(
    policy: RawPolicyContent,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> IngestionResult:
    if policy.status.casefold() != "current":
        return IngestionResult(
            status=IngestionStatus.FAILED,
            chunks=(),
            error="Policy is not current.",
        )

    normalized_text = normalize_policy_text(
        policy.raw_text
    )

    if not normalized_text:
        return IngestionResult(
            status=IngestionStatus.FAILED,
            chunks=(),
            error="Policy content was empty.",
        )

    normalized_content_units: list[PolicyContentUnit] = []

    for unit in policy.content_units:
        normalized_unit_text = normalize_policy_text(
            unit.text
        )

        if not normalized_unit_text:
            continue

        normalized_content_units.append(
            PolicyContentUnit(
                heading_path=unit.heading_path,
                text=normalized_unit_text,
            )
        )

    parsed_policy = ParsedPolicyContent(
        policy_id=policy.policy_id,
        title=policy.title,
        source_url=policy.source_url,
        status=policy.status,
        effective_date=policy.effective_date,
        review_date=policy.review_date,
        text=normalized_text,
        content_units=tuple(normalized_content_units),
    )

    try:
        chunks = chunk_policy(
            parsed_policy,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )
    except ValueError as exc:
        return IngestionResult(
            status=IngestionStatus.FAILED,
            chunks=(),
            error=str(exc),
        )

    if not chunks:
        return IngestionResult(
            status=IngestionStatus.FAILED,
            chunks=(),
            error="Policy produced no chunks.",
        )

    return IngestionResult(
        status=IngestionStatus.SUCCESS,
        chunks=chunks,
        error=None,
    )