"""
RAVIN local command-line interface.

This file is only the command-line interface for RAVIN. It is not the
RAG backend itself.

The reusable backend is RavinAnswerService.

Both the command-line interface and the FastAPI application should use
the same RavinAnswerService. FastAPI must not duplicate RAVIN's
retrieval, routing, evidence checking, generation, or grounding logic.


HOW THIS CLI WORKS
==================

When this script starts it:

1. Downloads the current policy documents.
2. Processes and chunks those policies.
3. Creates RavinAnswerService.
4. Builds the retrieval index and loads the configured models once.
5. Accepts questions from the command line.
6. Sends each question to:

       service.answer(question)

7. Prints the returned answer and sources.

The expensive setup work happens once when the application starts.
It must not happen again for every question.


HOW THE FASTAPI APPLICATION SHOULD WORK
=======================================

FastAPI should be a thin HTTP layer around the same
RavinAnswerService used here.

The intended structure is:

    FastAPI application starts
        -> acquire and process the current policy corpus
        -> create RavinAnswerService once
        -> keep that service available while the application runs

    Client sends POST /api/questions
        -> read the question from the request body
        -> call service.answer(question)
        -> convert the result to JSON
        -> return the JSON response

FastAPI should NOT:

- run this CLI script
- rebuild the retrieval index for each request
- reload the models for each request
- perform its own retrieval
- decide whether evidence is sufficient
- decide the answer behavior
- call the LLM directly
- perform separate grounding logic

Those responsibilities already belong to RavinAnswerService and the
backend components it uses.


EXPECTED FASTAPI REQUEST
========================

The endpoint should be:

    POST /api/questions

The request body should contain:

    {
        "question": "What happens when a student is not making satisfactory academic progress?"
    }


EXPECTED FASTAPI PROCESS
========================

The FastAPI endpoint should do approximately this:

    result = service.answer(
        request.question
    )

It should then map the IntegratedAnswerResult returned by the service
into the agreed API response structure.


EXPECTED GROUNDED RESPONSE
==========================

For a grounded answer, return:

    {
        "answer": "The grounded answer returned by RAVIN.",
        "grounded": true,
        "sources": [
            {
                "policy_id": "220",
                "title": "Academic Progression Review Policy",
                "heading": "Relevant Section",
                "url": "https://policies.latrobe.edu.au/..."
            }
        ]
    }


EXPECTED NO-EVIDENCE RESPONSE
=============================

If RavinAnswerService determines that the available policy evidence is
not sufficient, return the result produced by the service:

    {
        "answer": "I could not find sufficient evidence in the available policy sources.",
        "grounded": false,
        "sources": []
    }

FastAPI should not override that decision or ask an LLM to decide
whether an answer should be returned.


SUGGESTED FASTAPI LIFECYCLE
===========================

The RAVIN service should be created once when FastAPI starts.

Conceptually:

    application startup
        -> acquire policy chunks
        -> create_ravin_answer_service(...)
        -> store the returned service

    each POST /api/questions request
        -> validate request
        -> service.answer(question)
        -> serialize result
        -> return response

This avoids rebuilding embeddings, indexes, parsers, rerankers, and
models for every request.


SEPARATION OF RESPONSIBILITIES
==============================

RavinAnswerService is responsible for:

    retrieval
    evidence sufficiency
    question routing
    grounded generation
    citation validation
    claim grounding validation
    source selection

FastAPI is responsible for:

    receiving HTTP requests
    validating the API request format
    calling RavinAnswerService
    converting the result to JSON
    returning HTTP responses

The CLI is responsible for:

    accepting terminal input
    calling RavinAnswerService
    printing results

This separation means the CLI can later be removed without changing
the backend, and the FastAPI application can expose the same backend
to the user interface.
"""

import argparse
from backend.core.answer_quality_config import (
    load_answer_quality_config,
)
from backend.core.runtime_config_loader import (
    load_runtime_provider_config,
)
from backend.ingestion.acquisition import (
    PolicyLink,
    acquire_policy,
)
from backend.ingestion.models import (
    PolicyChunk,
)
from backend.ingestion.processor import (
    process_policy,
)
from backend.service.answer_service import (
    IntegratedAnswerResult,
    RavinAnswerService,
)
from backend.service.composition import (
    create_ravin_answer_service,
)

POLICIES = (
    (
        "208",
        "Academic Dress Policy",
    ),
    (
        "220",
        "Academic Progression Review Policy",
    ),
    (
        "76",
        "Academic Promotions Policy",
    ),
    (
        "420",
        "Academic Staff Qualifications Policy",
    ),
    (
        "169",
        "Admissions Policy",
    ),
    (
        "340",
        "Admissions Procedure",
    ),
)
_POLICY_BASE_URL = (
    "https://policies.latrobe.edu.au/"
    "document/view.php?id="
)

def acquire_current_policy_chunks(
) -> tuple[
    PolicyChunk,
    ...
]:
    all_chunks: list[
        PolicyChunk
    ] = []

    print(
        "=== ACQUIRING CURRENT POLICIES ==="
    )

    for policy_id, title in POLICIES:
        link = PolicyLink(
            policy_id=policy_id,
            title=title,
            url=(
                f"{_POLICY_BASE_URL}"
                f"{policy_id}"
            ),
        )

        policy = acquire_policy(
            link
        )

        result = process_policy(
            policy
        )

        if not result.chunks:
            raise RuntimeError(
                "Policy ingestion failed for "
                f"{policy_id}: {result.error}"
            )

        print(
            f"{policy.policy_id} "
            f"{policy.title} "
            f"-> {len(result.chunks)} chunks"
        )

        all_chunks.extend(
            result.chunks
        )

    if not all_chunks:
        raise RuntimeError(
            "No policy chunks were acquired."
        )

    print()
    print(
        "Total policy chunks -> "
        f"{len(all_chunks)}"
    )

    return tuple(
        all_chunks
    )

def build_ravin_service(
    chunks: tuple[
        PolicyChunk,
        ...
    ],
) -> RavinAnswerService:
    runtime_config = (
        load_runtime_provider_config()
    )

    quality_config = (
        load_answer_quality_config()
    )

    print()
    print(
        "=== RUNTIME CONFIGURATION ==="
    )

    print(
        "Generation provider -> "
        f"{runtime_config.generation.provider}"
    )

    print(
        "Generation model -> "
        f"{runtime_config.generation.model}"
    )

    print(
        "Answer quality status -> "
        f"{quality_config.status}"
    )

    print()
    print(
        "=== BUILDING RAVIN SERVICE ==="
    )

    service = (
        create_ravin_answer_service(
            chunks,
            runtime_config=runtime_config,
            answer_quality_config=(
                quality_config
            ),
        )
    )

    print(
        "RAVIN service ready."
    )

    return service

def print_answer(
    result: IntegratedAnswerResult,
) -> None:
    print()
    print(
        "=== RAVIN RESULT ==="
    )

    print(
        f"Behavior -> "
        f"{result.behavior.value}"
    )

    print(
        f"Grounded -> "
        f"{result.grounded}"
    )

    print()
    print(
        "Answer:"
    )

    print(
        result.answer
    )

    if not result.sources:
        print()
        print(
            "Sources -> none"
        )

        return

    print()
    print(
        "Sources:"
    )

    for index, source in enumerate(
        result.sources,
        start=1,
    ):
        print()
        print(
            f"{index}. "
            f"{source.title}"
        )

        print(
            f"   Policy ID -> "
            f"{source.policy_id}"
        )

        if source.heading:
            print(
                f"   Heading -> "
                f"{source.heading}"
            )

        print(
            f"   URL -> "
            f"{source.url}"
        )

def ask_question(
    service: RavinAnswerService,
    question: str,
) -> None:
    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    print()
    print(
        "Question:"
    )

    print(
        question
    )

    result = service.answer(
        question
    )

    print_answer(
        result
    )

def _parse_arguments(
) -> argparse.Namespace:
    parser = (
        argparse.ArgumentParser(
            description=(
                "Run the local RAVIN "
                "policy assistant."
            )
        )
    )

    parser.add_argument(
        "question",
        nargs="*",
        help=(
            "Optional question. If omitted, "
            "RAVIN runs interactively."
        ),
    )

    return parser.parse_args()

def main() -> None:
    print(
        "=== RAVIN LOCAL BACKEND ==="
    )

    args = _parse_arguments()

    chunks = (
        acquire_current_policy_chunks()
    )

    service = build_ravin_service(
        chunks
    )

    supplied_question = " ".join(
        args.question
    ).strip()

    if supplied_question:
        ask_question(
            service,
            supplied_question,
        )

        return

    print()
    print(
        "Enter a policy question."
    )

    print(
        "Press Enter on an empty line "
        "to exit."
    )

    while True:
        print()

        question = input(
            "RAVIN> "
        ).strip()

        if not question:
            break

        ask_question(
            service,
            question,
        )

if __name__ == "__main__":
    main()