"""
Shared application bootstrap for RAVIN interfaces.

This module prepares the production RAVIN service from the current
policy corpus.

Application adapters such as the command-line interface and FastAPI
should use this module instead of duplicating policy acquisition,
ingestion, or service-construction logic.

Typical startup flow:

    application starts
        -> acquire current policies
        -> process and chunk policies
        -> create RavinAnswerService
        -> keep the service for the application lifetime

FastAPI should create the service once during application startup.
Individual HTTP requests should call:

    service.answer(question)

FastAPI must not rebuild the policy corpus, retrieval index, models,
or RavinAnswerService for every request.
"""

from collections.abc import Callable
from dataclasses import dataclass

from backend.core.answer_quality_config import (
    AnswerQualityConfig,
)
from backend.core.provider_composition import (
    ProviderFactories,
)
from backend.core.runtime_config import (
    RuntimeProviderConfig,
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
from backend.retrieval.context import (
    ContextAssemblyConfig,
)
from backend.retrieval.production import (
    ProductionRetrievalConfig,
)
from backend.service.answer_service import (
    RavinAnswerService,
)
from backend.service.composition import (
    create_ravin_answer_service,
)

_POLICY_BASE_URL = (
    "https://policies.latrobe.edu.au/"
    "document/view.php?id="
)
CURRENT_POLICY_LINKS = (
    PolicyLink(
        policy_id="208",
        title="Academic Dress Policy",
        url=f"{_POLICY_BASE_URL}208",
    ),
    PolicyLink(
        policy_id="220",
        title="Academic Progression Review Policy",
        url=f"{_POLICY_BASE_URL}220",
    ),
    PolicyLink(
        policy_id="76",
        title="Academic Promotions Policy",
        url=f"{_POLICY_BASE_URL}76",
    ),
    PolicyLink(
        policy_id="420",
        title="Academic Staff Qualifications Policy",
        url=f"{_POLICY_BASE_URL}420",
    ),
    PolicyLink(
        policy_id="169",
        title="Admissions Policy",
        url=f"{_POLICY_BASE_URL}169",
    ),
    PolicyLink(
        policy_id="340",
        title="Admissions Procedure",
        url=f"{_POLICY_BASE_URL}340",
    ),
)

@dataclass(frozen=True)
class PolicyLoadProgress:
    """Summary emitted after one current policy is processed."""

    policy_id: str
    title: str
    chunk_count: int

PolicyLoadCallback = Callable[
    [PolicyLoadProgress],
    None,
]

def acquire_current_policy_chunks(
    *,
    timeout_seconds: float = 15.0,
    on_policy_loaded: (
        PolicyLoadCallback | None
    ) = None,
) -> tuple[
    PolicyChunk,
    ...
]:
    """
    Acquire and process the configured current policy corpus.

    Only policies accepted by the existing ingestion pipeline are
    included. Any failed policy stops startup rather than silently
    creating an incomplete production corpus.
    """

    if timeout_seconds <= 0:
        raise ValueError(
            "Policy acquisition timeout must be greater than zero."
        )

    chunks: list[
        PolicyChunk
    ] = []

    for link in CURRENT_POLICY_LINKS:
        raw_policy = acquire_policy(
            link,
            timeout_seconds=timeout_seconds,
        )

        result = process_policy(
            raw_policy
        )

        if not result.chunks:
            raise RuntimeError(
                "Policy ingestion failed for "
                f"{link.policy_id}: {result.error}"
            )

        chunks.extend(
            result.chunks
        )

        if on_policy_loaded is not None:
            on_policy_loaded(
                PolicyLoadProgress(
                    policy_id=link.policy_id,
                    title=link.title,
                    chunk_count=len(
                        result.chunks
                    ),
                )
            )

    if not chunks:
        raise RuntimeError(
            "No current policy chunks were acquired."
        )

    return tuple(
        chunks
    )

def create_current_policy_ravin_service(
    *,
    runtime_config: (
        RuntimeProviderConfig | None
    ) = None,
    provider_factories: (
        ProviderFactories | None
    ) = None,
    answer_quality_config: (
        AnswerQualityConfig | None
    ) = None,
    retrieval_config: (
        ProductionRetrievalConfig | None
    ) = None,
    context_config: (
        ContextAssemblyConfig | None
    ) = None,
    timeout_seconds: float = 15.0,
    on_policy_loaded: (
        PolicyLoadCallback | None
    ) = None,
) -> RavinAnswerService:
    """
    Create one production RAVIN service from the current policies.

    Application adapters should call this during startup and reuse
    the returned service for subsequent questions.
    """

    chunks = acquire_current_policy_chunks(
        timeout_seconds=timeout_seconds,
        on_policy_loaded=on_policy_loaded,
    )

    return create_ravin_answer_service(
        chunks,
        runtime_config=runtime_config,
        provider_factories=provider_factories,
        answer_quality_config=answer_quality_config,
        retrieval_config=retrieval_config,
        context_config=context_config,
    )