import pytest
from backend.ingestion.models import PolicyChunk
from backend.retrieval.context import (
    ContextAssemblyConfig,
    find_structural_neighbors,
)

def make_chunk(
    chunk_index: int,
    heading_path: tuple[str, ...],
    policy_id: str = "169",
) -> PolicyChunk:
    return PolicyChunk(
        policy_id=policy_id,
        policy_title="Admissions Policy",
        source_url=(
            "https://policies.latrobe.edu.au/"
            f"document/view.php?id={policy_id}"
        ),
        status="Current",
        effective_date=None,
        review_date=None,
        chunk_index=chunk_index,
        text=f"Policy chunk {chunk_index}.",
        heading_path=heading_path,
    )

def test_context_config_uses_expected_defaults():
    config = ContextAssemblyConfig()

    assert config.neighbor_window == 1
    assert config.max_context_chunks == 15

def test_context_config_rejects_negative_neighbor_window():
    with pytest.raises(
        ValueError,
        match=(
            "Context neighbour window cannot "
            "be negative."
        ),
    ):
        ContextAssemblyConfig(
            neighbor_window=-1,
        )

def test_context_config_rejects_invalid_max_context_chunks():
    with pytest.raises(
        ValueError,
        match=(
            "Maximum context chunks must be "
            "greater than zero."
        ),
    ):
        ContextAssemblyConfig(
            max_context_chunks=0,
        )

def test_find_structural_neighbors_returns_previous_and_next():
    heading = (
        "Section 5 - Policy Statement",
    )

    chunks = (
        make_chunk(6, heading),
        make_chunk(7, heading),
        make_chunk(8, heading),
    )

    neighbors = find_structural_neighbors(
        chunks,
        anchor=chunks[1],
        window=1,
    )

    assert [
        chunk.chunk_index
        for chunk in neighbors
    ] == [
        6,
        8,
    ]

def test_find_structural_neighbors_does_not_cross_heading_boundary():
    section_five = (
        "Section 5 - Policy Statement",
    )

    section_six = (
        "Section 6 - Procedures",
    )

    chunks = (
        make_chunk(
            7,
            section_five,
        ),
        make_chunk(
            8,
            section_six,
        ),
    )

    neighbors = find_structural_neighbors(
        chunks,
        anchor=chunks[0],
        window=1,
    )

    assert neighbors == ()

def test_find_structural_neighbors_does_not_cross_policy_boundary():
    heading = (
        "Section 5 - Policy Statement",
    )

    anchor = make_chunk(
        7,
        heading,
        policy_id="169",
    )

    other_policy = make_chunk(
        8,
        heading,
        policy_id="340",
    )

    neighbors = find_structural_neighbors(
        (
            anchor,
            other_policy,
        ),
        anchor=anchor,
        window=1,
    )

    assert neighbors == ()

def test_find_structural_neighbors_respects_window():
    heading = (
        "Section 5 - Policy Statement",
    )

    chunks = tuple(
        make_chunk(
            index,
            heading,
        )
        for index in range(5, 10)
    )

    neighbors = find_structural_neighbors(
        chunks,
        anchor=chunks[2],
        window=2,
    )

    assert [
        chunk.chunk_index
        for chunk in neighbors
    ] == [
        5,
        6,
        8,
        9,
    ]

def test_find_structural_neighbors_zero_window_returns_empty():
    heading = (
        "Section 5 - Policy Statement",
    )

    chunks = (
        make_chunk(7, heading),
        make_chunk(8, heading),
    )

    neighbors = find_structural_neighbors(
        chunks,
        anchor=chunks[0],
        window=0,
    )

    assert neighbors == ()

def test_find_structural_neighbors_rejects_negative_window():
    heading = (
        "Section 5 - Policy Statement",
    )

    chunk = make_chunk(
        7,
        heading,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Neighbour window cannot be negative."
        ),
    ):
        find_structural_neighbors(
            (chunk,),
            anchor=chunk,
            window=-1,
        )