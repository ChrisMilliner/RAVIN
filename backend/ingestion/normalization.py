"""
Normalize acquired policy text before chunk construction.

This module removes source-formatting noise and standardises text used
by later ingestion stages while preserving the policy meaning required
for evidence retrieval.

Normalization occurs before indexing so retrieval and grounding operate
on consistent text representations.
"""

import re

def normalize_policy_text(text: str) -> str:
    """Normalize policy whitespace without changing substantive text content.

    Runs of whitespace are collapsed and surrounding whitespace is removed
    so indexing operates on a consistent textual representation.
    """
    normalized = re.sub(r"\s+", " ", text)
    return normalized.strip()