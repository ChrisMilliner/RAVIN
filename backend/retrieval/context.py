from dataclasses import dataclass
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