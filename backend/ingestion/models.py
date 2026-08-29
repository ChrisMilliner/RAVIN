"""
Define the structured data models used during policy ingestion.

These models represent policy content as it progresses from acquired
source material to parsed content, retrieval chunks, and final
ingestion results. They retain status and provenance information needed
to diagnose failures and trace retrieved evidence back to its source.

The models contain ingestion state rather than retrieval or routing
decisions.
"""

from dataclasses import dataclass
from enum import Enum

class IngestionStatus(str, Enum):
    """Represent the success or failure state of a policy ingestion attempt.

    The explicit state allows ingestion callers to distinguish rejected
    policy material from successfully produced retrieval chunks.
    """

    SUCCESS = "success"
    FAILED = "failed"

@dataclass(frozen=True)
class PolicyContentUnit:
    """Represent one structurally grouped unit of policy text.

    The heading path records the policy hierarchy that applies to the text
    and is retained when downstream retrieval chunks are created.
    """

    heading_path: tuple[str, ...]
    text: str

@dataclass(frozen=True)
class RawPolicyContent:
    """Represent policy material exactly after source acquisition.

    The model combines policy identity, source metadata, raw page text, and
    any extracted structural content units before normalization occurs.
    """

    policy_id: str
    title: str
    source_url: str
    status: str
    effective_date: str | None
    review_date: str | None
    raw_text: str
    content_units: tuple[PolicyContentUnit, ...] = ()

@dataclass(frozen=True)
class ParsedPolicyContent:
    """Represent normalized policy material prepared for chunking.

    Source identity and metadata are preserved while text and structured
    content units contain the normalized forms consumed by the chunking
    stage.
    """

    policy_id: str
    title: str
    source_url: str
    status: str
    effective_date: str | None
    review_date: str | None
    text: str
    content_units: tuple[PolicyContentUnit, ...] = ()

@dataclass(frozen=True)
class PolicyChunk:
    """Represent one traceable unit of policy evidence used by retrieval.

    Each chunk retains policy identity, title, source URL, currency
    metadata, stable chunk position, text, and structural heading path so a
    retrieved result can be traced back to its source.
    """

    policy_id: str
    policy_title: str
    source_url: str
    status: str
    effective_date: str | None
    review_date: str | None
    chunk_index: int
    text: str
    heading_path: tuple[str, ...] = ()

@dataclass(frozen=True)
class IngestionResult:
    """Represent the controlled outcome of processing one policy.

    Successful results contain retrieval-ready chunks. Failed results
    contain no approved chunks and provide an explanatory error suitable
    for ingestion diagnostics.
    """

    status: IngestionStatus
    chunks: tuple[PolicyChunk, ...]
    error: str | None