import os
from collections.abc import Mapping
from backend.core.provider_registry import (
    CROSS_ENCODER_PROVIDER,
    SENTENCE_TRANSFORMER_PROVIDER,
    SPACY_PROVIDER,
    CROSS_ENCODER_ANSWERABILITY_PROVIDER,
)
from backend.core.runtime_config import (
    ProviderModelConfig,
    QuestionParserProviderConfig,
    RetrievalProviderConfig,
    RuntimeProviderConfig,
)

EMBEDDING_PROVIDER_ENV = (
    "RAVIN_EMBEDDING_PROVIDER"
)
EMBEDDING_MODEL_ENV = (
    "RAVIN_EMBEDDING_MODEL"
)
RERANKER_PROVIDER_ENV = (
    "RAVIN_RERANKER_PROVIDER"
)
RERANKER_MODEL_ENV = (
    "RAVIN_RERANKER_MODEL"
)
PRIMARY_PARSER_PROVIDER_ENV = (
    "RAVIN_PRIMARY_PARSER_PROVIDER"
)
PRIMARY_PARSER_MODEL_ENV = (
    "RAVIN_PRIMARY_PARSER_MODEL"
)

FALLBACK_PARSER_PROVIDER_ENV = (
    "RAVIN_FALLBACK_PARSER_PROVIDER"
)
FALLBACK_PARSER_MODEL_ENV = (
    "RAVIN_FALLBACK_PARSER_MODEL"
)
DEFAULT_RUNTIME_EMBEDDING_PROVIDER = (
    SENTENCE_TRANSFORMER_PROVIDER
)
DEFAULT_RUNTIME_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_RUNTIME_RERANKER_PROVIDER = (
    CROSS_ENCODER_PROVIDER
)
DEFAULT_RUNTIME_RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)
DEFAULT_RUNTIME_PRIMARY_PARSER_PROVIDER = (
    SPACY_PROVIDER
)
DEFAULT_RUNTIME_PRIMARY_PARSER_MODEL = (
    "en_core_web_sm"
)
DEFAULT_RUNTIME_FALLBACK_PARSER_PROVIDER = (
    SPACY_PROVIDER
)
DEFAULT_RUNTIME_FALLBACK_PARSER_MODEL = (
    "en_core_web_md"
)
ANSWERABILITY_PROVIDER_ENV = (
    "RAVIN_ANSWERABILITY_PROVIDER"
)
ANSWERABILITY_MODEL_ENV = (
    "RAVIN_ANSWERABILITY_MODEL"
)
DEFAULT_RUNTIME_ANSWERABILITY_PROVIDER = (
    CROSS_ENCODER_ANSWERABILITY_PROVIDER
)
DEFAULT_RUNTIME_ANSWERABILITY_MODEL = (
    "cross-encoder/qnli-electra-base"
)

def _read_setting(
    environ: Mapping[str, str],
    name: str,
    default: str,
) -> str:
    value = environ.get(
        name,
        default,
    ).strip()

    if not value:
        raise ValueError(
            f"{name} cannot be empty."
        )

    return value

def load_runtime_provider_config(
    environ: Mapping[str, str] | None = None,
) -> RuntimeProviderConfig:
    source = (
        os.environ
        if environ is None
        else environ
    )

    embedding = ProviderModelConfig(
        provider=_read_setting(
            source,
            EMBEDDING_PROVIDER_ENV,
            DEFAULT_RUNTIME_EMBEDDING_PROVIDER,
        ),
        model=_read_setting(
            source,
            EMBEDDING_MODEL_ENV,
            DEFAULT_RUNTIME_EMBEDDING_MODEL,
        ),
    )

    reranker = ProviderModelConfig(
        provider=_read_setting(
            source,
            RERANKER_PROVIDER_ENV,
            DEFAULT_RUNTIME_RERANKER_PROVIDER,
        ),
        model=_read_setting(
            source,
            RERANKER_MODEL_ENV,
            DEFAULT_RUNTIME_RERANKER_MODEL,
        ),
    )

    primary_parser = ProviderModelConfig(
        provider=_read_setting(
            source,
            PRIMARY_PARSER_PROVIDER_ENV,
            DEFAULT_RUNTIME_PRIMARY_PARSER_PROVIDER,
        ),
        model=_read_setting(
            source,
            PRIMARY_PARSER_MODEL_ENV,
            DEFAULT_RUNTIME_PRIMARY_PARSER_MODEL,
        ),
    )

    fallback_parser = ProviderModelConfig(
        provider=_read_setting(
            source,
            FALLBACK_PARSER_PROVIDER_ENV,
            DEFAULT_RUNTIME_FALLBACK_PARSER_PROVIDER,
        ),
        model=_read_setting(
            source,
            FALLBACK_PARSER_MODEL_ENV,
            DEFAULT_RUNTIME_FALLBACK_PARSER_MODEL,
        ),
    )

    answerability = ProviderModelConfig(
        provider=_read_setting(
            source,
            ANSWERABILITY_PROVIDER_ENV,
            DEFAULT_RUNTIME_ANSWERABILITY_PROVIDER,
        ),
        model=_read_setting(
            source,
            ANSWERABILITY_MODEL_ENV,
            DEFAULT_RUNTIME_ANSWERABILITY_MODEL,
        ),
    )

    return RuntimeProviderConfig(
        retrieval=RetrievalProviderConfig(
            embedding=embedding,
            reranker=reranker,
        ),
        question_parser=(
            QuestionParserProviderConfig(
                primary=primary_parser,
                fallback=fallback_parser,
            )
        ),
        answerability=answerability,
    )
