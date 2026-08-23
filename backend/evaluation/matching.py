from backend.evaluation.models import ExpectedEvidence
from backend.ingestion.models import PolicyChunk

def matches_expected_evidence(
    chunk: PolicyChunk,
    expected: ExpectedEvidence,
) -> bool:
    if chunk.policy_id != expected.policy_id:
        return False

    expected_path = expected.heading_path
    actual_path = chunk.heading_path

    if len(expected_path) > len(actual_path):
        return False

    return (
        actual_path[:len(expected_path)]
        == expected_path
    )