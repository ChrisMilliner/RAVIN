from backend.ingestion.models import (
    IngestionStatus,
    RawPolicyContent,
    PolicyContentUnit,
)
from backend.ingestion.processor import process_policy

def _make_policy(
    raw_text: str,
    status: str = "Current",
) -> RawPolicyContent:
    return RawPolicyContent(
        policy_id="208",
        title="Example Policy",
        source_url="https://example.invalid/document/view.php?id=208",
        status=status,
        effective_date="14 November 2025",
        review_date="13 November 2028",
        raw_text=raw_text,
    )

def test_current_policy_with_valid_text_is_ingested():
    result = process_policy(
        _make_policy(
            "Students may request an assessment extension.",
        )
    )

    assert result.status is IngestionStatus.SUCCESS
    assert len(result.chunks) == 1
    assert result.error is None

def test_empty_policy_content_is_rejected():
    result = process_policy(
        _make_policy(""),
    )

    assert result.status is IngestionStatus.FAILED
    assert result.chunks == ()
    assert result.error == "Policy content was empty."

def test_whitespace_only_policy_content_is_rejected():
    result = process_policy(
        _make_policy("   \n\t   "),
    )

    assert result.status is IngestionStatus.FAILED
    assert result.chunks == ()
    assert result.error == "Policy content was empty."

def test_non_current_policy_is_rejected():
    result = process_policy(
        _make_policy(
            "This policy has valid text.",
            status="Historic",
        )
    )

    assert result.status is IngestionStatus.FAILED
    assert result.chunks == ()
    assert result.error == "Policy is not current."

def test_future_policy_is_rejected():
    result = process_policy(
        _make_policy(
            "This policy contains future policy text.",
            status="Future",
        )
    )

    assert result.status is IngestionStatus.FAILED
    assert result.chunks == ()
    assert result.error == "Policy is not current."

def test_ingestion_normalizes_whitespace_before_chunking():
    result = process_policy(
        _make_policy(
            "Students    may\n\nrequest\tan extension.",
        )
    )

    assert result.status is IngestionStatus.SUCCESS
    assert result.chunks[0].text == (
        "Students may request an extension."
    )

def test_ingestion_preserves_provenance_in_chunks():
    result = process_policy(
        _make_policy(
            "Students may request an assessment extension.",
        )
    )

    chunk = result.chunks[0]

    assert chunk.policy_id == "208"
    assert chunk.policy_title == "Example Policy"
    assert (
        chunk.source_url
        == "https://example.invalid/document/view.php?id=208"
    )
    assert chunk.status == "Current"
    assert chunk.effective_date == "14 November 2025"
    assert chunk.review_date == "13 November 2028"

def test_ingestion_preserves_chunk_order():
    result = process_policy(
        _make_policy(
            "one two three four five six seven eight nine ten"
        ),
        chunk_size_words=4,
        overlap_words=1,
    )

    assert result.status is IngestionStatus.SUCCESS
    assert len(result.chunks) == 3

    assert result.chunks[0].chunk_index == 0
    assert result.chunks[0].text == "one two three four"

    assert result.chunks[1].chunk_index == 1
    assert result.chunks[1].text == "four five six seven"

    assert result.chunks[2].chunk_index == 2
    assert result.chunks[2].text == "seven eight nine ten"

def test_invalid_chunk_configuration_returns_failed_ingestion():
    result = process_policy(
        _make_policy(
            "Students may request an assessment extension.",
        ),
        chunk_size_words=10,
        overlap_words=10,
    )

    assert result.status is IngestionStatus.FAILED
    assert result.chunks == ()
    assert result.error == (
        "Chunk overlap must be smaller than chunk size."
    )

def test_all_successful_chunks_retain_current_status():
    result = process_policy(
        _make_policy(
            "one two three four five six seven eight nine ten"
        ),
        chunk_size_words=4,
        overlap_words=1,
    )

    assert result.status is IngestionStatus.SUCCESS

    assert all(
        chunk.status == "Current"
        for chunk in result.chunks
    )

def test_ingestion_preserves_heading_path_in_chunks():
    policy = RawPolicyContent(
        policy_id="TEST-STRUCTURE",
        title="Structured Policy",
        source_url="https://example.invalid/structured",
        status="Current",
        effective_date=None,
        review_date=None,
        raw_text="Section content.",
        content_units=(
            PolicyContentUnit(
                heading_path=(
                    "Section 6 - Procedures",
                    "Part A - Example",
                ),
                text="Structured section content.",
            ),
        ),
    )

    result = process_policy(
        policy,
        chunk_size_words=100,
        overlap_words=20,
    )

    assert result.status == IngestionStatus.SUCCESS

    assert result.chunks[0].heading_path == (
        "Section 6 - Procedures",
        "Part A - Example",
    )