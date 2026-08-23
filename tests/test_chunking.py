from backend.ingestion.chunking import chunk_policy
from backend.ingestion.models import (
    ParsedPolicyContent,
    PolicyContentUnit,
)

def make_policy(
    content_units: tuple[PolicyContentUnit, ...],
) -> ParsedPolicyContent:
    return ParsedPolicyContent(
        policy_id="TEST-001",
        title="Example Policy",
        source_url="https://example.invalid/policy",
        status="Current",
        effective_date=None,
        review_date=None,
        text="Fallback policy text.",
        content_units=content_units,
    )

def test_chunking_does_not_cross_structural_boundaries():
    policy = make_policy(
        (
            PolicyContentUnit(
                heading_path=("Section 1",),
                text="one two three four five six",
            ),
            PolicyContentUnit(
                heading_path=("Section 2",),
                text="seven eight nine ten eleven twelve",
            ),
        )
    )

    chunks = chunk_policy(
        policy,
        chunk_size_words=4,
        overlap_words=1,
    )

    section_one_chunks = [
        chunk
        for chunk in chunks
        if chunk.heading_path == ("Section 1",)
    ]

    section_two_chunks = [
        chunk
        for chunk in chunks
        if chunk.heading_path == ("Section 2",)
    ]

    assert section_one_chunks
    assert section_two_chunks

    assert all(
        "seven" not in chunk.text
        for chunk in section_one_chunks
    )

    assert all(
        "one" not in chunk.text
        for chunk in section_two_chunks
    )

def test_long_structural_unit_is_split_with_same_heading_path():
    policy = make_policy(
        (
            PolicyContentUnit(
                heading_path=(
                    "Section 6 - Procedures",
                    "Part A - Example",
                ),
                text=(
                    "one two three four five "
                    "six seven eight nine ten"
                ),
            ),
        )
    )

    chunks = chunk_policy(
        policy,
        chunk_size_words=4,
        overlap_words=1,
    )

    assert len(chunks) > 1

    assert all(
        chunk.heading_path
        == (
            "Section 6 - Procedures",
            "Part A - Example",
        )
        for chunk in chunks
    )

def test_structured_chunk_indices_remain_sequential():
    policy = make_policy(
        (
            PolicyContentUnit(
                heading_path=("Section 1",),
                text="one two three four five",
            ),
            PolicyContentUnit(
                heading_path=("Section 2",),
                text="six seven eight nine ten",
            ),
        )
    )

    chunks = chunk_policy(
        policy,
        chunk_size_words=4,
        overlap_words=1,
    )

    assert [
        chunk.chunk_index
        for chunk in chunks
    ] == list(range(len(chunks)))