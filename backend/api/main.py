"""
RAVIN grounded-answer API (COPF-231).

A standalone FastAPI service that accepts a policy question and returns a
grounded response, distinguishing between "sufficient evidence found" and
"insufficient evidence" outcomes. Usable independently via Swagger/OpenAPI
without requiring the frontend to exist.

Retrieval, evidence assessment, and response building are NOT reimplemented
here - this calls Chris's real backend.core.response.build_grounded_response,
which itself calls backend.core.retrieval.retrieve_evidence and
backend.core.evidence.assess_evidence. This satisfies COPF-231's
requirement to reuse existing foundations rather than duplicate them.

Security controls implemented here map to the team's API rules:
  Rule 3  - all input validated via Pydantic before use (models.py)
  Rule 4  - no secrets in code; generic error messages to clients
  Rule 6  - request-size limit (question length) + basic rate limiting
  Rule 7  - no personal/identifying data collected, only the question text
  Rule 9  - security-relevant events logged without logging full question
            content on every request
"""

import logging
import time
from collections import defaultdict, deque

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from backend.api.fixtures import API_POLICY_FIXTURES
from backend.api.models import AnswerResponse, QuestionRequest, SourceReference
from backend.core.models import ResponseOutcome
from backend.core.response import build_grounded_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ravin_api")

app = FastAPI(
    title="RAVIN Grounded-Answer API",
    description=(
        "Standalone service for the Policy DB Chatbot. Accepts a policy "
        "question and returns a grounded answer with sources, or an "
        "insufficient-evidence response. Calls the real "
        "backend.core.response.build_grounded_response pipeline - "
        "retrieval and evidence-gating logic are not duplicated here."
    ),
    version="0.2.0",
)

# --- Rate limiting (Rule 6) ---------------------------------------------
RATE_LIMIT_MAX_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60
_request_log: dict[str, deque] = defaultdict(deque)


def _is_rate_limited(client_id: str) -> bool:
    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    timestamps = _request_log[client_id]
    while timestamps and timestamps[0] < window_start:
        timestamps.popleft()
    if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        return True
    timestamps.append(now)
    return False


# --- Error handlers (Rule 4: never expose internals to the client) -----
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.info("Request validation failed for %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "invalid_request",
            "message": "The request could not be processed. Check the question field and try again.",
        },
    )


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    logger.info("Model validation failed for %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "invalid_request",
            "message": "The request could not be processed. Check the question field and try again.",
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    # backend.core.response.build_grounded_response raises ValueError for
    # whitespace-only questions - translate to a controlled 422, not a 500.
    logger.info("Core pipeline rejected input for %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "invalid_request",
            "message": "The request could not be processed. Check the question field and try again.",
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error processing request to %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_error",
            "message": "Something went wrong processing your request. Please try again.",
        },
    )


# --- Routes --------------------------------------------------------------
@app.get("/health", tags=["system"])
async def health_check():
    """Basic liveness check. Returns no sensitive information."""
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse, tags=["chatbot"])
async def ask_question(payload: QuestionRequest, request: Request):
    """
    Submit a policy question and receive a grounded answer.

    Calls backend.core.response.build_grounded_response directly - the
    same function Chris's demo.py uses - against API_POLICY_FIXTURES
    (his original 2 fixtures plus 4 additional sample policies for demo
    variety; see backend/api/fixtures.py).
    """
    client_id = request.client.host if request.client else "unknown"

    if _is_rate_limited(client_id):
        logger.warning("Rate limit exceeded for client %s", client_id)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "rate_limited",
                "message": "Too many requests. Please wait before trying again.",
            },
        )

    core_response = build_grounded_response(payload.question, API_POLICY_FIXTURES)

    sources = [
        SourceReference(
            policy_id=s.policy_id,
            title=s.policy_title,
            source_url=s.source_url,
            relevance_score=round(s.relevance_score, 4),
        )
        for s in core_response.sources
    ]

    return AnswerResponse(
        grounded=core_response.outcome is ResponseOutcome.SUPPORTED,
        answer=core_response.answer or "",
        sources=sources,
    )
