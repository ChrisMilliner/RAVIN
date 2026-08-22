from dataclasses import dataclass
from backend.ingestion.config import (
    DEFAULT_CHUNK_OVERLAP_WORDS,
)
from backend.ingestion.models import PolicyChunk

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