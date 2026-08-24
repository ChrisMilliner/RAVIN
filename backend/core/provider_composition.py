from dataclasses import dataclass
from typing import (
    Callable,
    Mapping,
    TypeVar,
)
from backend.core.runtime_config import (
    ProviderModelConfig,
    QuestionParserProviderConfig,
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
from backend.routing.answerability import (
    AnswerabilityProvider,
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
AnswerabilityProviderFactory = Callable[
    [str],
    AnswerabilityProvider,
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

    answerability: Mapping[
        str,
        AnswerabilityProviderFactory,
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

def compose_embedding_provider(
    config: ProviderModelConfig,
    factories: ProviderFactories,
) -> EmbeddingProvider:
    return _create_provider(
        config,
        factories.embedding,
        "embedding",
    )

def compose_reranker_provider(
    config: ProviderModelConfig,
    factories: ProviderFactories,
) -> RerankerProvider:
    return _create_provider(
        config,
        factories.reranker,
        "reranker",
    )

def compose_answerability_provider(
    config: ProviderModelConfig,
    factories: ProviderFactories,
) -> AnswerabilityProvider:
    return _create_provider(
        config,
        factories.answerability,
        "answerability",
    )

def compose_question_parser(
    config: QuestionParserProviderConfig,
    factories: ProviderFactories,
) -> QuestionParser:
    primary_parser = _create_provider(
        config.primary,
        factories.question_parser,
        "question parser",
    )

    fallback_config = config.fallback

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

    return QuestionParser(
        primary_provider=primary_parser,
        fallback_provider_factory=(
            fallback_factory
        ),
    )

def compose_runtime_providers(
    config: RuntimeProviderConfig,
    factories: ProviderFactories,
) -> ComposedProviders:
    return ComposedProviders(
        embedding=compose_embedding_provider(
            config.retrieval.embedding,
            factories,
        ),
        reranker=compose_reranker_provider(
            config.retrieval.reranker,
            factories,
        ),
        question_parser=compose_question_parser(
            config.question_parser,
            factories,
        ),
    )