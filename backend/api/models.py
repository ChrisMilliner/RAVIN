"""
HTTP request/response models for the RAVIN API (COPF-231).

Updated to wrap backend.service's real IntegratedAnswerResult /
AnswerSource (from RavinAnswerService.answer()), now that the real
service exists (merged via PR #9 / COPF-222). No longer wraps the
earlier backend.core.response.build_grounded_response fixture-based
pipeline.
"""

from pydantic import BaseModel, Field, field_validator

MAX_QUESTION_LENGTH = 500
MIN_QUESTION_LENGTH = 1


class QuestionRequest(BaseModel):
    """Incoming request: a single natural-language policy question."""

    question: str = Field(
        ...,
        description="A natural-language policy question from the user.",
        examples=["What is the current admissions policy?"],
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
    """A single supporting policy source, derived from backend.service.AnswerSource."""

    policy_id: str
    title: str
    heading: str
    url: str


class AnswerResponse(BaseModel):
    """
    HTTP response returned for every question, derived from
    backend.service.answer_service.IntegratedAnswerResult.

    `behavior` is the real routing outcome as a string: one of
    "direct_answer", "grounded_overview", "clarify", or
    "no_grounded_answer" (see backend.behavior.AnswerBehavior).

    `grounded` is True only for answer-producing behaviors
    (direct_answer / grounded_overview); False for clarify and
    no_grounded_answer.
    """

    behavior: str
    grounded: bool
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
