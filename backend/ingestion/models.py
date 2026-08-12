from dataclasses import dataclass
from enum import Enum

class IngestionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"

@dataclass(frozen=True)
class PolicyContentUnit:
    heading_path: tuple[str, ...]
    text: str

@dataclass(frozen=True)
class RawPolicyContent:
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
    status: IngestionStatus
    chunks: tuple[PolicyChunk, ...]
    error: str | None