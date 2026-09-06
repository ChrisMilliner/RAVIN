"""
RAVIN grounded-answer API (COPF-231).

Thin FastAPI adapter around the shared RavinAnswerService, following the
pattern from Chris's team handoff deck (RAVIN_Backend_Team_Handoff.pptx,
slides 9-10):

    FastAPI owns:              request validation, calling the shared
                                service, mapping the result to JSON
    RAVIN backend owns:        retrieval, evidence assessment, routing,
                                generation, validation

This module does not import retrieval, embedding, routing, or generation
components directly - it only ever calls service.answer(question).

Endpoint: POST /api/questions (the agreed Sprint 3 API contract, per
Chris's review of the earlier version of this PR).

Changes from the previous version of this file, per Chris's requested
changes on PR #10:
  1. Endpoint moved from /ask to POST /api/questions.
  2. service.answer() is synchronous; this route is now a plain `def`
     (not `async def`), so FastAPI runs it in its worker threadpool
     automatically rather than blocking the event loop.
  3. The broad `except ValueError -> 422` handler has been removed.
     RavinAnswerService.answer() raises ValueError only for
     whitespace-only input, which Pydantic validation (models.py)
     already rejects before the service is ever called - so a
     ValueError escaping from here now indicates a genuine backend
     fault and is treated as an internal error (500), not invalid user
     input. See COPF-240 for an example of exactly this kind of bug.

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
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from backend.api.models import AnswerResponse, QuestionRequest, SourceReference
from backend.service.bootstrap import create_current_policy_ravin_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ravin_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Build the RavinAnswerService once, at startup, and reuse it for the
    lifetime of the application (team handoff deck, slide 5: "Build
    once, reuse for every request").

    This performs real acquisition of the current policy corpus (live
    HTTP requests to policies.latrobe.edu.au) and constructs the full
    retrieval/routing/generation pipeline. Requires a local Ollama
    server to be running.
    """
    logger.info("Building RavinAnswerService from current policy corpus...")

    def _log_progress(progress):
        logger.info(
            "Loaded policy %s (%s): %d chunks",
            progress.policy_id,
            progress.title,
            progress.chunk_count,
        )

    app.state.ravin_service = create_current_policy_ravin_service(
        on_policy_loaded=_log_progress,
    )
    logger.info("RavinAnswerService ready.")

    yield


app = FastAPI(
    title="RAVIN Grounded-Answer API",
    description=(
        "Thin FastAPI adapter around the shared RavinAnswerService. "
        "Does not duplicate retrieval, routing, or generation logic."
    ),
    version="0.4.0",
    lifespan=lifespan,
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


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Deliberately broad and deliberately last-resort: this now includes
    # ValueError raised by RavinAnswerService itself (e.g. COPF-240-style
    # bugs), which are genuine backend faults, not invalid user input.
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
async def health_check(request: Request):
    """Basic liveness/readiness check. Returns no sensitive information."""
    ready = getattr(request.app.state, "ravin_service", None) is not None
    return {"status": "ok", "service_ready": ready}


@app.post("/api/questions", response_model=AnswerResponse, tags=["chatbot"])
def ask_question(payload: QuestionRequest, request: Request):
    """
    Submit a policy question and receive a routed, grounded answer.

    Note: this is a synchronous `def` route, not `async def`. FastAPI
    runs sync routes in a worker threadpool automatically, so the
    blocking call to service.answer() does not stall the event loop for
    other concurrent requests.
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

    service = request.app.state.ravin_service
    result = service.answer(payload.question)

    sources = [
        SourceReference(
            policy_id=s.policy_id,
            title=s.title,
            heading=s.heading,
            url=s.url,
        )
        for s in result.sources
    ]

    return AnswerResponse(
        behavior=result.behavior.value,
        grounded=result.grounded,
        answer=result.answer,
        sources=sources,
    )
