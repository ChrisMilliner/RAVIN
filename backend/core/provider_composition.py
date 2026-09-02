"""
Compose configured runtime providers behind framework-neutral contracts.

This module converts RAVIN runtime configuration into concrete provider
instances for embeddings, reranking, question parsing, answerability,
entailment, and grounded language generation.

Business logic depends on the neutral provider contracts rather than
specific model libraries or vendors. This keeps provider replacement
separate from retrieval, routing, generation, and validation logic.
"""

from dataclasses import (
    dataclass,
    field,
)
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
from backend.llm.provider import (
    LanguageModelProvider,
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
from backend.generation.entailment import (
    EntailmentProvider,
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
EntailmentProviderFactory = Callable[
    [str],
    EntailmentProvider,
]
LanguageModelProviderFactory = Callable[
    [str],
    LanguageModelProvider,
]

@dataclass(frozen=True)
class ProviderFactories:
    """Group the provider factories available during runtime composition.

    Factories isolate construction of concrete model and library adapters
    from the business components that consume their neutral contracts.
    """

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

    entailment: Mapping[
        str,
        EntailmentProviderFactory,
    ] = field(
        default_factory=dict
    )

    language_model: Mapping[
        str,
        LanguageModelProviderFactory,
    ] = field(
        default_factory=dict
    )

@dataclass(frozen=True)
class ComposedProviders:
    """Hold the concrete provider instances composed for one RAVIN runtime.

    The collection supplies retrieval, parsing, evidence-assessment,
    entailment, and generation providers to the application composition
    layer without exposing provider-selection logic to business services.
    """

    embedding: EmbeddingProvider
    reranker: RerankerProvider
    question_parser: QuestionParser
    answerability: AnswerabilityProvider
    entailment: EntailmentProvider
    language_model: LanguageModelProvider

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
    """Create the configured embedding provider.

    Provider selection is resolved at the composition boundary so semantic
    retrieval depends on the neutral embedding contract rather than a
    specific embedding library or model.
    """
    return _create_provider(
        config,
        factories.embedding,
        "embedding",
    )

def compose_reranker_provider(
    config: ProviderModelConfig,
    factories: ProviderFactories,
) -> RerankerProvider:
    """Create the configured retrieval reranker provider.

    The returned adapter implements the neutral reranking contract used by
    production retrieval.
    """
    return _create_provider(
        config,
        factories.reranker,
        "reranker",
    )

def compose_answerability_provider(
    config: ProviderModelConfig,
    factories: ProviderFactories,
) -> AnswerabilityProvider:
    """Create the configured proposition-answerability provider.

    This provider supplies evidence scores to proposition coverage
    assessment and does not independently determine final evidence
    sufficiency or answer behaviour.
    """
    return _create_provider(
        config,
        factories.answerability,
        "answerability",
    )

def compose_entailment_provider(
    config: ProviderModelConfig,
    factories: ProviderFactories,
) -> EntailmentProvider:
    """Create the configured natural-language-inference entailment provider.

    The provider is used to validate generated factual claims against their
    cited evidence during the fail-closed release process.
    """
    return _create_provider(
        config,
        factories.entailment,
        "entailment",
    )

def compose_language_model_provider(
    config: ProviderModelConfig,
    factories: ProviderFactories,
) -> LanguageModelProvider:
    """Create the configured language-model provider for grounded wording.

    The returned provider is used only after deterministic routing has
    approved grounded generation. It does not control intent, sufficiency,
    routing, or answer release.
    """
    return _create_provider(
        config,
        factories.language_model,
        "language model",
    )

def compose_question_parser(
    config: QuestionParserProviderConfig,
    factories: ProviderFactories,
) -> QuestionParser:
    """Create the configured question-parser service.

    The service combines the primary parser with an optional fallback while
    exposing RAVIN's framework-neutral question parsing interface to
    routing components.
    """
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
    """Compose all configured model and parser providers for one RAVIN runtime.

    This is the central provider-composition boundary used during
    application startup. Business services receive the resulting neutral
    provider interfaces rather than constructing model implementations
    themselves.
    """
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
        answerability=(
            compose_answerability_provider(
                config.answerability,
                factories,
            )
        ),
        entailment=(
            compose_entailment_provider(
                config.entailment,
                factories,
            )
        ),
        language_model=(
            compose_language_model_provider(
                config.generation,
                factories,
            )
        ),
    )
