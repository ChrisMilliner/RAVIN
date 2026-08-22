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

def find_structural_neighbors(
    chunks: tuple[PolicyChunk, ...],
    anchor: PolicyChunk,
    window: int = DEFAULT_NEIGHBOR_WINDOW,
) -> tuple[PolicyChunk, ...]:
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
