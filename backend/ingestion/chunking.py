"""
Split parsed policy content into retrieval-ready policy chunks.

Chunking preserves policy identity, heading structure, source location,
and the text needed for semantic and lexical retrieval. The resulting
PolicyChunk objects provide the stable evidence units indexed by the
retrieval layer.

Chunk construction does not perform semantic ranking or answer
generation.
"""

from backend.ingestion.models import (
    ParsedPolicyContent,
    PolicyChunk,
    PolicyContentUnit,
)

def _validate_chunk_configuration(
    chunk_size_words: int,
    overlap_words: int,
) -> None:
    if chunk_size_words <= 0:
        raise ValueError(
            "Chunk size must be greater than zero."
        )

    if overlap_words < 0:
        raise ValueError(
            "Chunk overlap cannot be negative."
        )

    if overlap_words >= chunk_size_words:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

def _split_text_into_word_chunks(
    text: str,
    chunk_size_words: int,
    overlap_words: int,
) -> tuple[str, ...]:
    words = text.split()

    if not words:
        return ()

    chunks: list[str] = []
    step = chunk_size_words - overlap_words

    for start in range(0, len(words), step):
        end = start + chunk_size_words

        chunk_words = words[start:end]

        if not chunk_words:
            break

        chunks.append(
            " ".join(chunk_words)
        )

        if end >= len(words):
            break

    return tuple(chunks)

def chunk_policy(
    policy: ParsedPolicyContent,
    chunk_size_words: int,
    overlap_words: int,
) -> tuple[PolicyChunk, ...]:
    """Split normalized policy content into overlapping retrieval chunks.

    Chunking validates the requested size and overlap, preserves heading
    paths and policy provenance, and assigns monotonically increasing chunk
    indexes across the policy.

    When no structured content units exist, the normalized policy text is
    treated as a single structural unit.
    """
    _validate_chunk_configuration(
        chunk_size_words,
        overlap_words,
    )

    chunks: list[PolicyChunk] = []
    chunk_index = 0

    if policy.content_units:
        units = policy.content_units
    else:
        units = (
            PolicyContentUnit(
                heading_path=(),
                text=policy.text,
            ),
        )

    for unit in units:
        text_chunks = _split_text_into_word_chunks(
            unit.text,
            chunk_size_words,
            overlap_words,
        )

        for text in text_chunks:
            chunks.append(
                PolicyChunk(
                    policy_id=policy.policy_id,
                    policy_title=policy.title,
                    source_url=policy.source_url,
                    status=policy.status,
                    effective_date=policy.effective_date,
                    review_date=policy.review_date,
                    chunk_index=chunk_index,
                    text=text,
                    heading_path=unit.heading_path,
                )
            )

            chunk_index += 1

    return tuple(chunks)