from dataclasses import FrozenInstanceError
import pytest
from backend.ingestion.models import (
    IngestionResult,
    IngestionStatus,
    ParsedPolicyContent,
    PolicyChunk,
    RawPolicyContent,
    PolicyContentUnit,
)

def test_raw_policy_content_preserves_source_metadata():
    policy = RawPolicyContent(
        policy_id="208",
        title="Example Policy",
        source_url="https://example.invalid/document/view.php?id=208",
        status="Current",
        effective_date="14 November 2025",
        review_date="13 November 2028",
        raw_text="Example raw policy text.",
    )

    assert policy.policy_id == "208"
    assert policy.title == "Example Policy"
    assert (
        policy.source_url
        == "https://example.invalid/document/view.php?id=208"
    )
    assert policy.status == "Current"
    assert policy.effective_date == "14 November 2025"
    assert policy.review_date == "13 November 2028"
    assert policy.raw_text == "Example raw policy text."

def test_raw_policy_content_is_immutable():
    policy = RawPolicyContent(
        policy_id="208",
        title="Example Policy",
        source_url="https://example.invalid/document/view.php?id=208",
        status="Current",
        effective_date="14 November 2025",
        review_date="13 November 2028",
        raw_text="Example raw policy text.",
    )

    with pytest.raises(FrozenInstanceError):
        setattr(policy, "title", "Changed Policy")

def test_parsed_policy_content_preserves_provenance():
    policy = ParsedPolicyContent(
        policy_id="208",
        title="Example Policy",
        source_url="https://example.invalid/document/view.php?id=208",
        status="Current",
        effective_date="14 November 2025",
        review_date="13 November 2028",
        text="Normalised policy text.",
    )

    assert policy.policy_id == "208"
    assert policy.title == "Example Policy"
    assert (
        policy.source_url
        == "https://example.invalid/document/view.php?id=208"
    )
    assert policy.status == "Current"
    assert policy.effective_date == "14 November 2025"
    assert policy.review_date == "13 November 2028"

def test_policy_chunk_preserves_provenance_and_position():
    chunk = PolicyChunk(
        policy_id="208",
        policy_title="Example Policy",
        source_url="https://example.invalid/document/view.php?id=208",
        status="Current",
        effective_date="14 November 2025",
        review_date="13 November 2028",
        chunk_index=2,
        text="A retrievable section of policy text.",
    )

    assert chunk.policy_id == "208"
    assert chunk.policy_title == "Example Policy"
    assert (
        chunk.source_url
        == "https://example.invalid/document/view.php?id=208"
    )
    assert chunk.status == "Current"
    assert chunk.effective_date == "14 November 2025"
    assert chunk.review_date == "13 November 2028"
    assert chunk.chunk_index == 2
    assert chunk.text == "A retrievable section of policy text."

def test_successful_ingestion_result_can_contain_chunks():
    chunk = PolicyChunk(
        policy_id="208",
        policy_title="Example Policy",
        source_url="https://example.invalid/document/view.php?id=208",
        status="Current",
        effective_date="14 November 2025",
        review_date="13 November 2028",
        chunk_index=0,
        text="A retrievable section of policy text.",
    )

    result = IngestionResult(
        status=IngestionStatus.SUCCESS,
        chunks=(chunk,),
        error=None,
    )

    assert result.status is IngestionStatus.SUCCESS
    assert result.chunks == (chunk,)
    assert result.error is None

def test_failed_ingestion_result_can_record_error_without_chunks():
    result = IngestionResult(
        status=IngestionStatus.FAILED,
        chunks=(),
        error="Policy content was empty.",
    )

    assert result.status is IngestionStatus.FAILED
    assert result.chunks == ()
    assert result.error == "Policy content was empty."

def test_policy_content_unit_preserves_heading_hierarchy():
    unit = PolicyContentUnit(
        heading_path=(
            "Section 6 - Procedures",
            "Part B - Requests",
        ),
        text="Policy procedure content.",
    )

    assert unit.heading_path == (
        "Section 6 - Procedures",
        "Part B - Requests",
    )
    assert unit.text == "Policy procedure content."