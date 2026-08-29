"""
HTTP request/response models for the RAVIN API (COPF-231).

These wrap backend.core's dataclasses (GroundedResponse, RetrievedEvidence)
into Pydantic models so FastAPI can validate requests and serialise
responses to JSON. The core dataclasses themselves are untouched -
conversion happens in backend/api/main.py.
"""

from pydantic import BaseModel, Field, field_validator

MAX_QUESTION_LENGTH = 500
MIN_QUESTION_LENGTH = 1


class QuestionRequest(BaseModel):
    """Incoming request: a single natural-language policy question."""

    question: str = Field(
        ...,
        description="A natural-language policy question from the user.",
        examples=["What is the policy on assignment extensions?"],
    )

    @field_validator("question")
    @classmethod
    def question_must_be_usable(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < MIN_QUESTION_LENGTH:
            raise ValueError("question must not be empty or whitespace-only")
        if len(stripped) > MAX_QUESTION_LENGTH:
            raise ValueError(
                f"question exceeds maximum length of {MAX_QUESTION_LENGTH} characters"
            )
        return stripped


class SourceReference(BaseModel):
    """A single supporting policy source, derived from RetrievedEvidence."""

    policy_id: str
    title: str
    source_url: str
    relevance_score: float


class AnswerResponse(BaseModel):
    """
    HTTP response returned for every question, derived from Chris's
    GroundedResponse (backend.core.models).

    `grounded` is True when outcome == ResponseOutcome.SUPPORTED,
    False when outcome == ResponseOutcome.INSUFFICIENT_EVIDENCE.
    """

    grounded: bool
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
