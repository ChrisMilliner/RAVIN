from dataclasses import dataclass
from typing import (
    Callable,
    Mapping,
    TypeVar,
)
from backend.core.runtime_config import (
    ProviderModelConfig,
    RuntimeProviderConfig,
)
from backend.retrieval.embeddings import (
    EmbeddingProvider,
)
from backend.retrieval.reranking import (
    RerankerProvider,
)
from backend.routing.question_parser import (
    QuestionParseProvider,
    QuestionParser,
)

EmbeddingProviderFactory = Callable[
    [str],
    EmbeddingProvider,
]
RerankerProviderFactory = Callable[
    [str],
    RerankerProvider,
]
QuestionParseProviderFactory = Callable[
    [str],
    QuestionParseProvider,
]

@dataclass(frozen=True)
class ProviderFactories:
    embedding: Mapping[
        str,
        EmbeddingProviderFactory,
    ]

    reranker: Mapping[
        str,
        RerankerProviderFactory,
    ]

    question_parser: Mapping[
        str,
        QuestionParseProviderFactory,
    ]

@dataclass(frozen=True)
class ComposedProviders:
    embedding: EmbeddingProvider
    reranker: RerankerProvider
    question_parser: QuestionParser

ProviderType = TypeVar(
    "ProviderType"
)

def _create_provider(
    config: ProviderModelConfig,
    factories: Mapping[
        str,
        Callable[
            [str],
            ProviderType,
        ],
    ],
    provider_kind: str,
) -> ProviderType:
    factory = factories.get(
        config.provider
    )

    if factory is None:
        raise ValueError(
            f"Unknown {provider_kind} provider: "
            f"{config.provider}."
        )

    return factory(
        config.model
    )

def compose_runtime_providers(
    config: RuntimeProviderConfig,
    factories: ProviderFactories,
) -> ComposedProviders:
    embedding = _create_provider(
        config.retrieval.embedding,
        factories.embedding,
        "embedding",
    )

    reranker = _create_provider(
        config.retrieval.reranker,
        factories.reranker,
        "reranker",
    )

    primary_parser = _create_provider(
        config.question_parser.primary,
        factories.question_parser,
        "question parser",
    )

    fallback_config = (
        config.question_parser.fallback
    )

    fallback_factory = None

    if fallback_config is not None:
        def create_fallback(
        ) -> QuestionParseProvider:
            return _create_provider(
                fallback_config,
                factories.question_parser,
                "question parser",
            )

        fallback_factory = create_fallback

    question_parser = QuestionParser(
        primary_provider=primary_parser,
        fallback_provider_factory=(
            fallback_factory
        ),
    )

    return ComposedProviders(
        embedding=embedding,
        reranker=reranker,
        question_parser=question_parser,
    )