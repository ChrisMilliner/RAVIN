"""
Build the textual representation used for policy retrieval.

This module combines relevant policy chunk content and structural
metadata into the text supplied to embedding, lexical, and reranking
operations.

Keeping retrieval-text construction explicit makes the indexed
representation reproducible and independently testable.
"""

from backend.ingestion.models import PolicyChunk

def build_retrieval_text(
    chunk: PolicyChunk,
) -> str:
    """Build the full text representation used for lexical scoring and reranking.

    The representation combines policy title, each heading in the structural
    path, and the policy chunk body in that order.
    """
    parts = [chunk.policy_title]

    parts.extend(chunk.heading_path)
    parts.append(chunk.text)

    return "\n".join(parts)