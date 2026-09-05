"""
RAVIN grounded-answer API (COPF-231) - updated to use the real
RavinAnswerService (merged via PR #9 / COPF-222), replacing the earlier
fixture-based build_grounded_response wrapper.

Per backend.service.bootstrap's own docstring:
    "FastAPI should create the service once during application startup.
    Individual HTTP requests should call: service.answer(question)
    FastAPI must not rebuild the policy corpus, retrieval index, models,
    or RavinAnswerService for every request."

This module follows that exactly: the service is built once in the
FastAPI lifespan startup hook (real network calls to the current La
Trobe policy pages happen here, once, not per-request), then reused for
every /ask request.

IMPORTANT: starting this app makes real HTTP requests to
policies.latrobe.edu.au (see backend.service.bootstrap.CURRENT_POLICY_LINKS).
Startup will take longer than before and requires real internet access -
this could not be pre-tested in the development sandbox this file was
written in, since that environment cannot reach that domain. Test this
locally before relying on it.

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
    lifetime of the application - per backend.service.bootstrap's
    explicit usage contract.

    This performs real acquisition of the current policy corpus (live
    HTTP requests to policies.latrobe.edu.au) and constructs the full
    retrieval/routing/generation pipeline. It can take noticeably longer
    than a typical app startup, and requires the local Ollama server to
    be running (see backend.llm.ollama_provider).
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

    # No explicit teardown required - the service holds no external
    # connections that need closing beyond the process lifetime.


app = FastAPI(
    title="RAVIN Grounded-Answer API",
    description=(
        "Standalone service for the Policy DB Chatbot. Accepts a policy "
        "question and returns a grounded answer with sources, or a "
        "clarify / no-grounded-answer response. Calls the real "
        "RavinAnswerService (backend.service.answer_service) built once "
        "at startup from the current policy corpus - no retrieval, "
        "routing, or generation logic is duplicated here."
    ),
    version="0.3.0",
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


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    # RavinAnswerService.answer() raises ValueError for whitespace-only
    # questions - translate to a controlled 422, not a 500.
    logger.info("Service rejected input for %s", request.url.path)
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
async def health_check(request: Request):
    """
    Basic liveness/readiness check. Returns no sensitive information.
    Reports whether the RavinAnswerService finished initialising.
    """
    ready = getattr(request.app.state, "ravin_service", None) is not None
    return {"status": "ok", "service_ready": ready}


@app.post("/ask", response_model=AnswerResponse, tags=["chatbot"])
async def ask_question(payload: QuestionRequest, request: Request):
    """
    Submit a policy question and receive a routed, grounded answer.

    Delegates entirely to the shared RavinAnswerService built at startup
    - this endpoint does no retrieval, routing, or generation itself.
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
