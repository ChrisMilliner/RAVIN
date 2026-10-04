"""
Shared application bootstrap for RAVIN interfaces.

This module prepares the production RAVIN service from the current
policy corpus.

Normal startup prefers a validated local retrieval cache containing the
processed policy corpus and precomputed document embeddings. When no
compatible cache exists, the live policy library is discovered,
acquired, processed, indexed, and cached.

A forced refresh deliberately rebuilds the complete accessible current
policy corpus from the live La Trobe Policy Library.

Application adapters such as the command-line interface and FastAPI
should use this module instead of duplicating policy acquisition,
ingestion, caching, or service-construction logic.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from backend.core.answer_quality_config import (
    AnswerQualityConfig,
)
from backend.core.provider_composition import (
    ProviderFactories,
)
from backend.core.runtime_config import (
    RuntimeProviderConfig,
)
from backend.core.runtime_config_loader import (
    load_runtime_provider_config,
)
from backend.ingestion.acquisition import (
    BROWSE_URL,
    PolicyAccessRestrictedError,
    acquire_policy,
    discover_policy_links,
    fetch_html,
)
from backend.ingestion.models import (
    PolicyChunk,
)
from backend.ingestion.processor import (
    process_policy,
)
from backend.retrieval.cache import (
    RetrievalCacheStaleError,
    load_retrieval_cache,
    save_retrieval_cache,
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

logger = logging.getLogger(
    "ravin_bootstrap"
)

RESTRICTED_POLICY_IDS = (
    "268",
)

DEFAULT_RETRIEVAL_CACHE_PATH = (
    Path(__file__)
    .resolve()
    .parents[2]
    / "data"
    / "ravin-retrieval-cache-v1.json.gz"
)

@dataclass(frozen=True)
class PolicyLoadProgress:
    """Summary emitted after one discovered policy is handled."""

    policy_id: str
    title: str
    chunk_count: int
    status: str

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
    Acquire and process the accessible current policy corpus.

    The live policy browse page is used to discover documents. Current
    accessible policies are processed into retrieval chunks. Known
    restricted policies are reported explicitly rather than silently
    disappearing from the corpus.

    Unexpected acquisition or ingestion failures stop startup so RAVIN
    does not silently construct an incomplete production corpus.
    """
    if timeout_seconds <= 0:
        raise ValueError(
            "Policy acquisition timeout must be greater than zero."
        )

    browse_html = fetch_html(
        BROWSE_URL,
        timeout_seconds=timeout_seconds,
    )

    policy_links = discover_policy_links(
        browse_html
    )

    if not policy_links:
        raise RuntimeError(
            "No policy links were discovered."
        )

    chunks: list[
        PolicyChunk
    ] = []

    restricted_policy_ids: list[
        str
    ] = []

    for link in policy_links:
        try:
            raw_policy = acquire_policy(
                link,
                timeout_seconds=timeout_seconds,
            )

        except PolicyAccessRestrictedError:
            restricted_policy_ids.append(
                link.policy_id
            )

            if on_policy_loaded is not None:
                on_policy_loaded(
                    PolicyLoadProgress(
                        policy_id=link.policy_id,
                        title=link.title,
                        chunk_count=0,
                        status="restricted",
                    )
                )

            continue

        result = process_policy(
            raw_policy
        )

        if not result.chunks:
            if (
                result.error
                == "Policy is not current."
            ):
                if on_policy_loaded is not None:
                    on_policy_loaded(
                        PolicyLoadProgress(
                            policy_id=link.policy_id,
                            title=link.title,
                            chunk_count=0,
                            status="not_current",
                        )
                    )

                continue

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
                    status="current",
                )
            )

    if not chunks:
        raise RuntimeError(
            "No current policy chunks were acquired."
        )

    unexpected_restricted = tuple(
        policy_id
        for policy_id
        in restricted_policy_ids
        if policy_id
        not in RESTRICTED_POLICY_IDS
    )

    if unexpected_restricted:
        raise RuntimeError(
            "Unexpected restricted policies: "
            + ", ".join(
                unexpected_restricted
            )
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
    cache_path: (
        Path | None
    ) = None,
    refresh_cache: bool = False,
) -> RavinAnswerService:
    """
    Create one production RAVIN service from the current policy corpus.

    Normal startup loads a compatible validated retrieval cache when
    available. A missing or embedding-model-incompatible cache triggers
    a complete live corpus build followed by cache creation.

    Cache corruption or structural validation failures are not silently
    ignored. They propagate to the caller so the operator can explicitly
    request a controlled refresh.

    refresh_cache=True bypasses any existing cache and deliberately
    rebuilds it from the live policy library.
    """
    resolved_runtime_config = (
        load_runtime_provider_config()
        if runtime_config is None
        else runtime_config
    )

    resolved_cache_path = (
        DEFAULT_RETRIEVAL_CACHE_PATH
        if cache_path is None
        else cache_path
    )

    embedding_config = (
        resolved_runtime_config
        .retrieval
        .embedding
    )

    if not refresh_cache:
        cache_existed = (
            resolved_cache_path.exists()
        )

        try:
            cached_corpus = (
                load_retrieval_cache(
                    resolved_cache_path,
                    embedding_provider=(
                        embedding_config.provider
                    ),
                    embedding_model=(
                        embedding_config.model
                    ),
                )
            )

        except RetrievalCacheStaleError as error:
            raise RuntimeError(
                (
                    "The local RAVIN policy cache is older "
                    "than 24 hours. Run "
                    "Start-RAVIN.ps1 -RefreshPolicyCache "
                    "to rebuild it from the live policy library."
                )
            ) from error

        if cached_corpus is not None:
            logger.info(
                (
                    "Loaded retrieval cache: "
                    "%d policies, %d chunks, "
                    "created %s"
                ),
                (
                    cached_corpus
                    .metadata
                    .policy_count
                ),
                (
                    cached_corpus
                    .metadata
                    .chunk_count
                ),
                (
                    cached_corpus
                    .metadata
                    .created_at_utc
                ),
            )

            cached_chunks = tuple(
                indexed_chunk.chunk
                for indexed_chunk
                in cached_corpus.indexed_chunks
            )

            return create_ravin_answer_service(
                cached_chunks,
                runtime_config=(
                    resolved_runtime_config
                ),
                provider_factories=(
                    provider_factories
                ),
                answer_quality_config=(
                    answer_quality_config
                ),
                retrieval_config=(
                    retrieval_config
                ),
                context_config=(
                    context_config
                ),
                indexed_chunks=(
                    cached_corpus
                    .indexed_chunks
                ),
            )

        if cache_existed:
            logger.info(
                (
                    "Existing retrieval cache is "
                    "incompatible with the configured "
                    "embedding provider/model. "
                    "Rebuilding it."
                )
            )
        else:
            logger.info(
                (
                    "Retrieval cache not found. "
                    "Building the current policy corpus."
                )
            )

    else:
        logger.info(
            (
                "Retrieval cache refresh requested. "
                "Rebuilding the current policy corpus."
            )
        )

    progress: list[
        PolicyLoadProgress
    ] = []

    def report_progress(
        item: PolicyLoadProgress,
    ) -> None:
        progress.append(
            item
        )

        if on_policy_loaded is not None:
            on_policy_loaded(
                item
            )

    chunks = acquire_current_policy_chunks(
        timeout_seconds=timeout_seconds,
        on_policy_loaded=report_progress,
    )

    restricted_policy_ids = tuple(
        item.policy_id
        for item in progress
        if item.status
        == "restricted"
    )

    def cache_completed_index(
        indexed_chunks,
    ) -> None:
        metadata = save_retrieval_cache(
            resolved_cache_path,
            indexed_chunks,
            embedding_provider=(
                embedding_config.provider
            ),
            embedding_model=(
                embedding_config.model
            ),
            restricted_policy_ids=(
                restricted_policy_ids
            ),
        )

        logger.info(
            (
                "Saved retrieval cache: "
                "%d policies, %d chunks, "
                "restricted policies=%s"
            ),
            metadata.policy_count,
            metadata.chunk_count,
            (
                ", ".join(
                    metadata
                    .restricted_policy_ids
                )
                or "none"
            ),
        )

    return create_ravin_answer_service(
        chunks,
        runtime_config=(
            resolved_runtime_config
        ),
        provider_factories=(
            provider_factories
        ),
        answer_quality_config=(
            answer_quality_config
        ),
        retrieval_config=(
            retrieval_config
        ),
        context_config=(
            context_config
        ),
        on_index_ready=(
            cache_completed_index
        ),
    )