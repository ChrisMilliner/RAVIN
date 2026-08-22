import pytest
from backend.ingestion.models import PolicyChunk
from backend.retrieval.context import (
    ContextAssemblyConfig,
    assemble_context_chunks,
    find_structural_neighbors,
    merge_structural_chunk_text,
    GroundedContextBlock,
    build_grounded_context_blocks,
)
from backend.retrieval.models import RetrievalResult

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

def make_result(
    chunk: PolicyChunk,
    score: float,
) -> RetrievalResult:
    return RetrievalResult(
        chunk=chunk,
        score=score,
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

def test_merge_structural_chunk_text_removes_expected_overlap():
    heading = (
        "Section 5 - Policy Statement",
    )

    chunks = (
        PolicyChunk(
            policy_id="169",
            policy_title="Admissions Policy",
            source_url="https://example.invalid/169",
            status="Current",
            effective_date=None,
            review_date=None,
            chunk_index=0,
            text="one two three four",
            heading_path=heading,
        ),
        PolicyChunk(
            policy_id="169",
            policy_title="Admissions Policy",
            source_url="https://example.invalid/169",
            status="Current",
            effective_date=None,
            review_date=None,
            chunk_index=1,
            text="three four five six",
            heading_path=heading,
        ),
    )

    merged = merge_structural_chunk_text(
        chunks,
        overlap_words=2,
    )

    assert merged == (
        "one two three four five six"
    )

def test_merge_structural_chunk_text_handles_multiple_chunks():
    heading = (
        "Section 5 - Policy Statement",
    )

    chunks = (
        PolicyChunk(
            policy_id="169",
            policy_title="Admissions Policy",
            source_url="https://example.invalid/169",
            status="Current",
            effective_date=None,
            review_date=None,
            chunk_index=0,
            text="one two three four",
            heading_path=heading,
        ),
        PolicyChunk(
            policy_id="169",
            policy_title="Admissions Policy",
            source_url="https://example.invalid/169",
            status="Current",
            effective_date=None,
            review_date=None,
            chunk_index=1,
            text="four five six seven",
            heading_path=heading,
        ),
        PolicyChunk(
            policy_id="169",
            policy_title="Admissions Policy",
            source_url="https://example.invalid/169",
            status="Current",
            effective_date=None,
            review_date=None,
            chunk_index=2,
            text="seven eight nine ten",
            heading_path=heading,
        ),
    )

    merged = merge_structural_chunk_text(
        chunks,
        overlap_words=1,
    )

    assert merged == (
        "one two three four five six seven "
        "eight nine ten"
    )

def test_merge_structural_chunk_text_keeps_nonmatching_words():
    heading = (
        "Section 5 - Policy Statement",
    )

    chunks = (
        PolicyChunk(
            policy_id="169",
            policy_title="Admissions Policy",
            source_url="https://example.invalid/169",
            status="Current",
            effective_date=None,
            review_date=None,
            chunk_index=0,
            text="one two three four",
            heading_path=heading,
        ),
        PolicyChunk(
            policy_id="169",
            policy_title="Admissions Policy",
            source_url="https://example.invalid/169",
            status="Current",
            effective_date=None,
            review_date=None,
            chunk_index=1,
            text="four five six seven",
            heading_path=heading,
        ),
    )

    merged = merge_structural_chunk_text(
        chunks,
        overlap_words=2,
    )

    assert merged == (
        "one two three four four five six seven"
    )

def test_merge_structural_chunk_text_zero_overlap_keeps_all_text():
    heading = (
        "Section 5 - Policy Statement",
    )

    chunks = (
        make_chunk(0, heading),
        make_chunk(1, heading),
    )

    merged = merge_structural_chunk_text(
        chunks,
        overlap_words=0,
    )

    assert merged == (
        "Policy chunk 0. Policy chunk 1."
    )

def test_merge_structural_chunk_text_rejects_heading_change():
    chunks = (
        make_chunk(
            0,
            ("Section 5",),
        ),
        make_chunk(
            1,
            ("Section 6",),
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Context chunks must form a "
            "consecutive structural sequence."
        ),
    ):
        merge_structural_chunk_text(
            chunks,
            overlap_words=1,
        )

def test_merge_structural_chunk_text_rejects_nonconsecutive_chunks():
    heading = (
        "Section 5",
    )

    chunks = (
        make_chunk(0, heading),
        make_chunk(2, heading),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Context chunks must form a "
            "consecutive structural sequence."
        ),
    ):
        merge_structural_chunk_text(
            chunks,
            overlap_words=1,
        )

def test_merge_structural_chunk_text_rejects_policy_change():
    heading = (
        "Section 5",
    )

    chunks = (
        make_chunk(
            0,
            heading,
            policy_id="169",
        ),
        make_chunk(
            1,
            heading,
            policy_id="340",
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Context chunks must form a "
            "consecutive structural sequence."
        ),
    ):
        merge_structural_chunk_text(
            chunks,
            overlap_words=1,
        )

def test_merge_structural_chunk_text_rejects_negative_overlap():
    heading = (
        "Section 5",
    )

    chunk = make_chunk(
        0,
        heading,
    )

    with pytest.raises(
        ValueError,
        match="Chunk overlap cannot be negative.",
    ):
        merge_structural_chunk_text(
            (chunk,),
            overlap_words=-1,
        )

def test_merge_structural_chunk_text_empty_input_returns_empty():
    assert merge_structural_chunk_text(
        (),
    ) == ""

def test_assemble_context_chunks_keeps_retrieved_seeds_first():
    heading = (
        "Section 5",
    )

    chunks = tuple(
        make_chunk(
            index,
            heading,
        )
        for index in range(5)
    )

    retrieval_results = (
        make_result(
            chunks[3],
            0.95,
        ),
        make_result(
            chunks[1],
            0.90,
        ),
    )

    assembled = assemble_context_chunks(
        chunks,
        retrieval_results,
        ContextAssemblyConfig(
            neighbor_window=1,
            max_context_chunks=6,
        ),
    )

    assert [
        chunk.chunk_index
        for chunk in assembled[:2]
    ] == [
        3,
        1,
    ]

def test_assemble_context_chunks_adds_structural_neighbors():
    heading = (
        "Section 5",
    )

    chunks = tuple(
        make_chunk(
            index,
            heading,
        )
        for index in range(5)
    )

    retrieval_results = (
        make_result(
            chunks[2],
            0.95,
        ),
    )

    assembled = assemble_context_chunks(
        chunks,
        retrieval_results,
        ContextAssemblyConfig(
            neighbor_window=1,
            max_context_chunks=5,
        ),
    )

    assert [
        chunk.chunk_index
        for chunk in assembled
    ] == [
        2,
        1,
        3,
    ]

def test_assemble_context_chunks_deduplicates_seed_neighbors():
    heading = (
        "Section 5",
    )

    chunks = tuple(
        make_chunk(
            index,
            heading,
        )
        for index in range(4)
    )

    retrieval_results = (
        make_result(
            chunks[1],
            0.95,
        ),
        make_result(
            chunks[2],
            0.90,
        ),
    )

    assembled = assemble_context_chunks(
        chunks,
        retrieval_results,
        ContextAssemblyConfig(
            neighbor_window=1,
            max_context_chunks=6,
        ),
    )

    assert [
        chunk.chunk_index
        for chunk in assembled
    ] == [
        1,
        2,
        0,
        3,
    ]

def test_assemble_context_chunks_deduplicates_retrieved_seeds():
    heading = (
        "Section 5",
    )

    chunk = make_chunk(
        1,
        heading,
    )

    retrieval_results = (
        make_result(
            chunk,
            0.95,
        ),
        make_result(
            chunk,
            0.90,
        ),
    )

    assembled = assemble_context_chunks(
        (chunk,),
        retrieval_results,
        ContextAssemblyConfig(
            neighbor_window=0,
            max_context_chunks=5,
        ),
    )

    assert assembled == (
        chunk,
    )

def test_assemble_context_chunks_respects_maximum():
    heading = (
        "Section 5",
    )

    chunks = tuple(
        make_chunk(
            index,
            heading,
        )
        for index in range(7)
    )

    retrieval_results = (
        make_result(
            chunks[3],
            0.95,
        ),
        make_result(
            chunks[5],
            0.90,
        ),
    )

    assembled = assemble_context_chunks(
        chunks,
        retrieval_results,
        ContextAssemblyConfig(
            neighbor_window=2,
            max_context_chunks=4,
        ),
    )

    assert len(assembled) == 4

    assert [
        chunk.chunk_index
        for chunk in assembled[:2]
    ] == [
        3,
        5,
    ]

def test_assemble_context_chunks_does_not_cross_heading_boundary():
    first_heading = (
        "Section 5",
    )

    second_heading = (
        "Section 6",
    )

    chunks = (
        make_chunk(
            0,
            first_heading,
        ),
        make_chunk(
            1,
            second_heading,
        ),
    )

    retrieval_results = (
        make_result(
            chunks[0],
            0.95,
        ),
    )

    assembled = assemble_context_chunks(
        chunks,
        retrieval_results,
        ContextAssemblyConfig(
            neighbor_window=1,
            max_context_chunks=5,
        ),
    )

    assert assembled == (
        chunks[0],
    )

def test_assemble_context_chunks_rejects_limit_below_seed_count():
    heading = (
        "Section 5",
    )

    chunks = tuple(
        make_chunk(
            index,
            heading,
        )
        for index in range(3)
    )

    retrieval_results = (
        make_result(
            chunks[0],
            0.95,
        ),
        make_result(
            chunks[1],
            0.90,
        ),
        make_result(
            chunks[2],
            0.85,
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Maximum context chunks cannot be "
            "smaller than the retrieved seed count."
        ),
    ):
        assemble_context_chunks(
            chunks,
            retrieval_results,
            ContextAssemblyConfig(
                neighbor_window=1,
                max_context_chunks=2,
            ),
        )

def test_assemble_context_chunks_is_deterministic():
    heading = (
        "Section 5",
    )

    chunks = tuple(
        make_chunk(
            index,
            heading,
        )
        for index in range(7)
    )

    retrieval_results = (
        make_result(
            chunks[3],
            0.95,
        ),
        make_result(
            chunks[5],
            0.90,
        ),
    )

    config = ContextAssemblyConfig(
        neighbor_window=2,
        max_context_chunks=6,
    )

    first = assemble_context_chunks(
        chunks,
        retrieval_results,
        config,
    )

    second = assemble_context_chunks(
        chunks,
        retrieval_results,
        config,
    )

    assert first == second

def test_grounded_context_block_preserves_provenance():
    block = GroundedContextBlock(
        policy_id="169",
        policy_title="Admissions Policy",
        source_url="https://example.invalid/169",
        heading_path=(
            "Section 5",
        ),
        start_chunk_index=7,
        end_chunk_index=9,
        text="Grounded policy evidence.",
    )

    assert block.policy_id == "169"
    assert block.policy_title == (
        "Admissions Policy"
    )
    assert block.source_url == (
        "https://example.invalid/169"
    )
    assert block.heading_path == (
        "Section 5",
    )
    assert block.start_chunk_index == 7
    assert block.end_chunk_index == 9
    assert block.text == (
        "Grounded policy evidence."
    )

def test_build_grounded_context_blocks_merges_consecutive_chunks():
    heading = (
        "Section 5",
    )

    chunks = (
        PolicyChunk(
            policy_id="169",
            policy_title="Admissions Policy",
            source_url="https://example.invalid/169",
            status="Current",
            effective_date=None,
            review_date=None,
            chunk_index=1,
            text="three four five six",
            heading_path=heading,
        ),
        PolicyChunk(
            policy_id="169",
            policy_title="Admissions Policy",
            source_url="https://example.invalid/169",
            status="Current",
            effective_date=None,
            review_date=None,
            chunk_index=0,
            text="one two three four",
            heading_path=heading,
        ),
    )

    blocks = build_grounded_context_blocks(
        chunks,
        overlap_words=2,
    )

    assert len(blocks) == 1

    block = blocks[0]

    assert block.start_chunk_index == 0
    assert block.end_chunk_index == 1
    assert block.text == (
        "one two three four five six"
    )

def test_build_grounded_context_blocks_separates_headings():
    chunks = (
        make_chunk(
            0,
            ("Section 5",),
        ),
        make_chunk(
            1,
            ("Section 6",),
        ),
    )

    blocks = build_grounded_context_blocks(
        chunks,
    )

    assert len(blocks) == 2

    assert blocks[0].heading_path == (
        "Section 5",
    )

    assert blocks[1].heading_path == (
        "Section 6",
    )

def test_build_grounded_context_blocks_separates_nonconsecutive_chunks():
    heading = (
        "Section 5",
    )

    chunks = (
        make_chunk(
            0,
            heading,
        ),
        make_chunk(
            2,
            heading,
        ),
    )

    blocks = build_grounded_context_blocks(
        chunks,
    )

    assert len(blocks) == 2

    assert blocks[0].start_chunk_index == 0
    assert blocks[0].end_chunk_index == 0

    assert blocks[1].start_chunk_index == 2
    assert blocks[1].end_chunk_index == 2

def test_build_grounded_context_blocks_preserves_selection_priority():
    first_heading = (
        "Section 6",
    )

    second_heading = (
        "Section 5",
    )

    chunks = (
        make_chunk(
            10,
            first_heading,
        ),
        make_chunk(
            2,
            second_heading,
        ),
    )

    blocks = build_grounded_context_blocks(
        chunks,
    )

    assert blocks[0].start_chunk_index == 10
    assert blocks[1].start_chunk_index == 2

def test_build_grounded_context_blocks_handles_seed_first_neighbor_order():
    heading = (
        "Section 5",
    )

    chunks = (
        PolicyChunk(
            policy_id="169",
            policy_title="Admissions Policy",
            source_url="https://example.invalid/169",
            status="Current",
            effective_date=None,
            review_date=None,
            chunk_index=2,
            text="five six seven eight",
            heading_path=heading,
        ),
        PolicyChunk(
            policy_id="169",
            policy_title="Admissions Policy",
            source_url="https://example.invalid/169",
            status="Current",
            effective_date=None,
            review_date=None,
            chunk_index=1,
            text="three four five six",
            heading_path=heading,
        ),
        PolicyChunk(
            policy_id="169",
            policy_title="Admissions Policy",
            source_url="https://example.invalid/169",
            status="Current",
            effective_date=None,
            review_date=None,
            chunk_index=3,
            text="seven eight nine ten",
            heading_path=heading,
        ),
    )

    blocks = build_grounded_context_blocks(
        chunks,
        overlap_words=2,
    )

    assert len(blocks) == 1

    assert blocks[0].start_chunk_index == 1
    assert blocks[0].end_chunk_index == 3
    assert blocks[0].text == (
        "three four five six seven eight "
        "nine ten"
    )

def test_build_grounded_context_blocks_rejects_duplicates():
    heading = (
        "Section 5",
    )

    chunk = make_chunk(
        1,
        heading,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Grounded context cannot contain "
            "duplicate policy chunks."
        ),
    ):
        build_grounded_context_blocks(
            (
                chunk,
                chunk,
            ),
        )


def test_build_grounded_context_blocks_empty_input_returns_empty():
    assert build_grounded_context_blocks(
        (),
    ) == ()