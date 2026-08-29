"""
Match retrieved policy evidence against expected evaluation evidence.

This module applies the structured policy, heading, descendant, and
text requirements defined by evaluation questions to determine whether
a retrieved result covers the expected evidence.

Evidence matching provides the basis for retrieval metrics and grounded
overview coverage calculations.
"""

from backend.evaluation.models import (
    ExpectedEvidence,
    ExpectedEvidenceGroup,
)
from backend.ingestion.models import PolicyChunk

def _normalize_text(value: str) -> str:
    return " ".join(
        value.split()
    ).casefold()

def matches_expected_evidence(
    chunk: PolicyChunk,
    expected: ExpectedEvidence,
) -> bool:
    if chunk.policy_id != expected.policy_id:
        return False

    expected_path = expected.heading_path
    actual_path = chunk.heading_path

    if expected.allow_descendants:
        if len(expected_path) > len(actual_path):
            return False

        heading_matches = (
            actual_path[:len(expected_path)]
            == expected_path
        )
    else:
        heading_matches = (
            actual_path == expected_path
        )

    if not heading_matches:
        return False

    if expected.text_contains is None:
        return True

    return (
        _normalize_text(expected.text_contains)
        in _normalize_text(chunk.text)
    )

def is_expected_evidence_group_covered(
    chunks: tuple[PolicyChunk, ...],
    group: ExpectedEvidenceGroup,
) -> bool:
    return any(
        matches_expected_evidence(
            chunk,
            alternative,
        )
        for chunk in chunks
        for alternative in group.alternatives
    )
