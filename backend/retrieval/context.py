"""
Assemble retrieved policy chunks into traceable grounded context.

This module expands selected retrieval results with appropriate
structural neighbours, groups related evidence, and renders labelled
context blocks for downstream evidence assessment and generation.

Context assembly preserves policy and heading provenance and is bounded
so additional neighbouring text does not grow the supplied evidence
without control.
"""

from dataclasses import dataclass
from backend.ingestion.config import (
    DEFAULT_CHUNK_OVERLAP_WORDS,
)
from backend.ingestion.models import PolicyChunk
from backend.retrieval.models import RetrievalResult

DEFAULT_NEIGHBOR_WINDOW = 1
DEFAULT_MAX_CONTEXT_CHUNKS = 15

@dataclass(frozen=True)
class ContextAssemblyConfig:
    """Configure bounded structural expansion of retrieved policy context.

    neighbor_window controls how far context assembly may search around
    each retrieved chunk within the same policy heading. max_context_chunks
    places an upper bound on the total selected context.
    """

    neighbor_window: int = (
        DEFAULT_NEIGHBOR_WINDOW
    )
    max_context_chunks: int = (
        DEFAULT_MAX_CONTEXT_CHUNKS
    )

    def __post_init__(self) -> None:
        if self.neighbor_window < 0:
            raise ValueError(
                "Context neighbour window cannot "
                "be negative."
            )

        if self.max_context_chunks <= 0:
            raise ValueError(
                "Maximum context chunks must be "
                "greater than zero."
            )

@dataclass(frozen=True)
class GroundedContextBlock:
    """Represent one consecutive structural block of grounded policy evidence.

    A block belongs to one policy and heading path and records the inclusive
    chunk range merged into its text. The retained policy metadata allows
    the block to remain traceable to its source.
    """

    policy_id: str
    policy_title: str
    source_url: str
    heading_path: tuple[str, ...]
    start_chunk_index: int
    end_chunk_index: int
    text: str

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError(
                "Grounded context policy ID cannot "
                "be empty."
            )

        if not self.policy_title.strip():
            raise ValueError(
                "Grounded context policy title cannot "
                "be empty."
            )

        if not self.source_url.strip():
            raise ValueError(
                "Grounded context source URL cannot "
                "be empty."
            )

        if self.start_chunk_index < 0:
            raise ValueError(
                "Grounded context start chunk index "
                "cannot be negative."
            )

        if (
            self.end_chunk_index
            < self.start_chunk_index
        ):
            raise ValueError(
                "Grounded context end chunk index "
                "cannot be smaller than the start "
                "chunk index."
            )

        if not self.text.strip():
            raise ValueError(
                "Grounded context text cannot "
                "be empty."
            )

@dataclass(frozen=True)
class GroundedContext:
    """Contain the ordered evidence blocks assembled for a grounded question.

    The blocks provide the structured evidence supplied to downstream
    evidence assessment and grounded answer generation.
    """

    blocks: tuple[
        GroundedContextBlock,
        ...
    ]

    @property
    def evidence_count(self) -> int:
        """Return the number of grounded evidence blocks in this context.
        """
        return len(self.blocks)

def find_structural_neighbors(
    chunks: tuple[PolicyChunk, ...],
    anchor: PolicyChunk,
    window: int = DEFAULT_NEIGHBOR_WINDOW,
) -> tuple[PolicyChunk, ...]:
    """Find nearby chunks that share the anchor policy and heading structure.

    Only chunks within the configured index distance are returned, ordered
    by chunk index. The anchor itself is excluded and a negative window is
    rejected.
    """
    if window < 0:
        raise ValueError(
            "Neighbour window cannot be negative."
        )

    if window == 0:
        return ()

    neighbors = tuple(
        chunk
        for chunk in chunks
        if (
            chunk.policy_id == anchor.policy_id
            and chunk.heading_path
            == anchor.heading_path
            and chunk.chunk_index
            != anchor.chunk_index
            and abs(
                chunk.chunk_index
                - anchor.chunk_index
            )
            <= window
        )
    )

    return tuple(
        sorted(
            neighbors,
            key=lambda chunk: chunk.chunk_index,
        )
    )

def merge_structural_chunk_text(
    chunks: tuple[PolicyChunk, ...],
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> str:
    """Merge a consecutive structural chunk sequence into one evidence text.

    All chunks must belong to the same policy and heading and have
    consecutive indexes. Expected chunk overlap is removed when present so
    the resulting evidence block does not repeat duplicated boundary text.
    """
    if overlap_words < 0:
        raise ValueError(
            "Chunk overlap cannot be negative."
        )

    if not chunks:
        return ""

    for previous, current in zip(
        chunks,
        chunks[1:],
    ):
        if (
            previous.policy_id != current.policy_id
            or previous.heading_path
            != current.heading_path
            or current.chunk_index
            != previous.chunk_index + 1
        ):
            raise ValueError(
                "Context chunks must form a "
                "consecutive structural sequence."
            )

    merged_words = chunks[0].text.split()

    for previous, current in zip(
        chunks,
        chunks[1:],
    ):
        previous_words = previous.text.split()
        current_words = current.text.split()

        has_expected_overlap = (
            overlap_words > 0
            and len(previous_words)
            >= overlap_words
            and len(current_words)
            >= overlap_words
            and previous_words[-overlap_words:]
            == current_words[:overlap_words]
        )

        if has_expected_overlap:
            merged_words.extend(
                current_words[overlap_words:]
            )
        else:
            merged_words.extend(
                current_words
            )

    return " ".join(merged_words)

def assemble_context_chunks(
    chunks: tuple[PolicyChunk, ...],
    retrieval_results: tuple[
        RetrievalResult,
        ...
    ],
    config: ContextAssemblyConfig,
) -> tuple[PolicyChunk, ...]:
    """Expand ranked retrieval seeds with bounded structural neighbours.

    Retrieved seed chunks are preserved first in retrieval order. Additional
    same-heading neighbours are then added by increasing distance until the
    configured context limit is reached.

    Duplicate chunks are excluded and the configured maximum cannot be
    smaller than the number of unique retrieval seeds.
    """
    selected: list[PolicyChunk] = []
    selected_keys: set[tuple[str, int]] = set()

    seed_chunks: list[PolicyChunk] = []

    for result in retrieval_results:
        chunk = result.chunk
        key = (
            chunk.policy_id,
            chunk.chunk_index,
        )

        if key in selected_keys:
            continue

        seed_chunks.append(chunk)
        selected.append(chunk)
        selected_keys.add(key)

    if len(seed_chunks) > config.max_context_chunks:
        raise ValueError(
            "Maximum context chunks cannot be "
            "smaller than the retrieved seed count."
        )

    if (
        len(selected)
        >= config.max_context_chunks
        or config.neighbor_window == 0
    ):
        return tuple(selected)

    for distance in range(
        1,
        config.neighbor_window + 1,
    ):
        for seed in seed_chunks:
            neighbors = find_structural_neighbors(
                chunks,
                anchor=seed,
                window=distance,
            )

            exact_distance_neighbors = (
                neighbor
                for neighbor in neighbors
                if abs(
                    neighbor.chunk_index
                    - seed.chunk_index
                )
                == distance
            )

            for neighbor in exact_distance_neighbors:
                key = (
                    neighbor.policy_id,
                    neighbor.chunk_index,
                )

                if key in selected_keys:
                    continue

                selected.append(neighbor)
                selected_keys.add(key)

                if (
                    len(selected)
                    >= config.max_context_chunks
                ):
                    return tuple(selected)

    return tuple(selected)

def build_grounded_context_blocks(
    chunks: tuple[PolicyChunk, ...],
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> tuple[GroundedContextBlock, ...]:
    """Convert selected policy chunks into ordered grounded evidence blocks.

    Chunks are grouped by policy provenance and heading path, split into
    consecutive runs, and merged with expected overlap removed. Block order
    retains the earliest selected position of each run so retrieval
    priority remains visible after structural grouping.
    """
    if overlap_words < 0:
        raise ValueError(
            "Chunk overlap cannot be negative."
        )

    if not chunks:
        return ()

    positions: dict[
        tuple[str, int],
        int,
    ] = {}

    for position, chunk in enumerate(chunks):
        key = (
            chunk.policy_id,
            chunk.chunk_index,
        )

        if key in positions:
            raise ValueError(
                "Grounded context cannot contain "
                "duplicate policy chunks."
            )

        positions[key] = position

    structural_groups: dict[
        tuple[
            str,
            str,
            str,
            tuple[str, ...],
        ],
        list[PolicyChunk],
    ] = {}

    for chunk in chunks:
        structural_key = (
            chunk.policy_id,
            chunk.policy_title,
            chunk.source_url,
            chunk.heading_path,
        )

        structural_groups.setdefault(
            structural_key,
            [],
        ).append(chunk)

    ranked_blocks: list[
        tuple[int, GroundedContextBlock]
    ] = []

    for group_chunks in structural_groups.values():
        ordered_chunks = sorted(
            group_chunks,
            key=lambda chunk: chunk.chunk_index,
        )

        runs: list[list[PolicyChunk]] = []
        current_run: list[PolicyChunk] = []

        for chunk in ordered_chunks:
            if not current_run:
                current_run.append(chunk)
                continue

            previous = current_run[-1]

            if (
                chunk.chunk_index
                == previous.chunk_index + 1
            ):
                current_run.append(chunk)
            else:
                runs.append(current_run)
                current_run = [chunk]

        if current_run:
            runs.append(current_run)

        for run in runs:
            run_tuple = tuple(run)

            block = GroundedContextBlock(
                policy_id=run[0].policy_id,
                policy_title=run[0].policy_title,
                source_url=run[0].source_url,
                heading_path=run[0].heading_path,
                start_chunk_index=(
                    run[0].chunk_index
                ),
                end_chunk_index=(
                    run[-1].chunk_index
                ),
                text=merge_structural_chunk_text(
                    run_tuple,
                    overlap_words=overlap_words,
                ),
            )

            first_selected_position = min(
                positions[
                    (
                        chunk.policy_id,
                        chunk.chunk_index,
                    )
                ]
                for chunk in run
            )

            ranked_blocks.append(
                (
                    first_selected_position,
                    block,
                )
            )

    ranked_blocks.sort(
        key=lambda item: item[0],
    )

    return tuple(
        block
        for _, block in ranked_blocks
    )

def build_grounded_context(
    chunks: tuple[PolicyChunk, ...],
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> GroundedContext:
    """Build a GroundedContext from selected policy chunks.

    The function delegates structural grouping and overlap-aware text
    merging to build_grounded_context_blocks.
    """
    return GroundedContext(
        blocks=build_grounded_context_blocks(
            chunks,
            overlap_words=overlap_words,
        )
    )

def render_grounded_context(
    context: GroundedContext,
) -> str:
    """Render grounded evidence blocks into labelled text for downstream use.

    Each block receives an E-number and includes policy identity, title,
    heading path, chunk range, source URL, and evidence text. These labels
    allow generated factual claims to reference specific supplied evidence.
    """
    rendered_blocks: list[str] = []

    for position, block in enumerate(
        context.blocks,
        start=1,
    ):
        evidence_id = f"E{position}"

        if block.heading_path:
            heading = " > ".join(
                block.heading_path
            )
        else:
            heading = "(document root)"

        if (
            block.start_chunk_index
            == block.end_chunk_index
        ):
            chunk_range = str(
                block.start_chunk_index
            )
        else:
            chunk_range = (
                f"{block.start_chunk_index}-"
                f"{block.end_chunk_index}"
            )

        rendered_blocks.append(
            "\n".join(
                (
                    f"[{evidence_id}]",
                    (
                        "Policy ID: "
                        f"{block.policy_id}"
                    ),
                    (
                        "Policy Title: "
                        f"{block.policy_title}"
                    ),
                    f"Heading: {heading}",
                    f"Chunks: {chunk_range}",
                    (
                        "Source: "
                        f"{block.source_url}"
                    ),
                    "Evidence:",
                    block.text,
                )
            )
        )

    return "\n\n".join(
        rendered_blocks
    )