"""
HOW FASTAPI SHOULD USE RAVIN
============================

FastAPI must use the same shared backend bootstrap as this CLI.

The shared startup function is:

    backend.service.bootstrap.create_current_policy_ravin_service

FastAPI should create the RAVIN service ONCE when the application
starts:

    service = create_current_policy_ravin_service()

That startup call:

    1. Acquires the configured current policies.
    2. Processes and chunks the policies.
    3. Loads the configured providers and models.
    4. Builds the retrieval index.
    5. Constructs RavinAnswerService.
    6. Returns the ready-to-use service.

FastAPI should keep that returned service for the lifetime of the
application.

For each:

    POST /api/questions

FastAPI should:

    1. Read and validate the question from the JSON request.
    2. Call:

           result = service.answer(request.question)

    3. Convert IntegratedAnswerResult into the agreed JSON response.
    4. Return that response.

FastAPI must NOT:

    - execute this CLI
    - acquire policies for every request
    - rebuild RavinAnswerService for every request
    - rebuild the retrieval index for every request
    - load models for every request
    - perform its own retrieval
    - decide evidence sufficiency
    - decide answer behaviour
    - call the LLM directly
    - perform separate citation or grounding validation

Those responsibilities already belong to the RAVIN backend.

The intended architecture is:

    current policy sources
        -> shared bootstrap
        -> RavinAnswerService
             |
             +-> CLI
             |
             +-> FastAPI

The CLI is responsible for terminal input and output.

FastAPI is responsible for HTTP input and output.

RavinAnswerService is responsible for RAVIN.
"""

import argparse
from backend.core.answer_quality_config import (
    load_answer_quality_config,
)
from backend.core.runtime_config_loader import (
    load_runtime_provider_config,
)
from backend.service.answer_service import (
    IntegratedAnswerResult,
    RavinAnswerService,
)
from backend.service.bootstrap import (
    PolicyLoadProgress,
    create_current_policy_ravin_service,
)

def build_ravin_service() -> RavinAnswerService:
    """
    Build the shared production RAVIN service for this CLI.

    Policy acquisition, ingestion, chunking, model composition, and
    retrieval-index construction are performed through the shared
    backend bootstrap also intended for the FastAPI application.
    """

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

    loaded_chunk_counts: list[int] = []

    def report_policy_loaded(
        progress: PolicyLoadProgress,
    ) -> None:
        loaded_chunk_counts.append(
            progress.chunk_count
        )

        print(
            f"{progress.policy_id} "
            f"{progress.title} -> "
            f"{progress.chunk_count} chunks"
        )

    print()
    print(
        "=== STARTING RAVIN SERVICE ==="
    )

    print()
    print(
        "Acquiring current policies:"
    )

    service = (
        create_current_policy_ravin_service(
            runtime_config=runtime_config,
            answer_quality_config=(
                quality_config
            ),
            on_policy_loaded=(
                report_policy_loaded
            ),
        )
    )

    print()

    print(
        "Total policy chunks -> "
        f"{sum(loaded_chunk_counts)}"
    )

    print()
    print(
        "RAVIN service ready."
    )

    return service

def print_answer(
    result: IntegratedAnswerResult,
) -> None:
    """Print an integrated RAVIN answer and its policy sources.
    """
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
    """Submit one question to a reusable RAVIN service and print the result.
    """
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
    """Run the local RAVIN CLI using the shared production service bootstrap.
    """
    print(
        "=== RAVIN LOCAL BACKEND ==="
    )

    args = _parse_arguments()

    service = build_ravin_service()

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