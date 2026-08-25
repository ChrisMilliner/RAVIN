import pytest
from backend.core.provider_registry import (
    CROSS_ENCODER_PROVIDER,
    SENTENCE_TRANSFORMER_PROVIDER,
    SPACY_PROVIDER,
    CROSS_ENCODER_ANSWERABILITY_PROVIDER,
)
from backend.core.runtime_config_loader import (
    DEFAULT_RUNTIME_EMBEDDING_MODEL,
    DEFAULT_RUNTIME_FALLBACK_PARSER_MODEL,
    DEFAULT_RUNTIME_PRIMARY_PARSER_MODEL,
    DEFAULT_RUNTIME_RERANKER_MODEL,
    EMBEDDING_MODEL_ENV,
    EMBEDDING_PROVIDER_ENV,
    FALLBACK_PARSER_MODEL_ENV,
    FALLBACK_PARSER_PROVIDER_ENV,
    PRIMARY_PARSER_MODEL_ENV,
    PRIMARY_PARSER_PROVIDER_ENV,
    RERANKER_MODEL_ENV,
    RERANKER_PROVIDER_ENV,
    ANSWERABILITY_MODEL_ENV,
    ANSWERABILITY_PROVIDER_ENV,
    DEFAULT_RUNTIME_ANSWERABILITY_MODEL,
    load_runtime_provider_config,
)

def test_loads_current_default_runtime_configuration():
    config = load_runtime_provider_config(
        {}
    )

    assert (
        config.retrieval.embedding.provider
        == SENTENCE_TRANSFORMER_PROVIDER
    )

    assert (
        config.retrieval.embedding.model
        == DEFAULT_RUNTIME_EMBEDDING_MODEL
    )

    assert (
        config.retrieval.reranker.provider
        == CROSS_ENCODER_PROVIDER
    )

    assert (
        config.retrieval.reranker.model
        == DEFAULT_RUNTIME_RERANKER_MODEL
    )

    assert (
        config.question_parser.primary.provider
        == SPACY_PROVIDER
    )

    assert (
        config.question_parser.primary.model
        == DEFAULT_RUNTIME_PRIMARY_PARSER_MODEL
    )

    assert (
        config.question_parser.fallback
        is not None
    )

    assert (
        config.question_parser.fallback.provider
        == SPACY_PROVIDER
    )

    assert (
        config.question_parser.fallback.model
        == DEFAULT_RUNTIME_FALLBACK_PARSER_MODEL
    )

    assert (
    config.answerability.provider
    == CROSS_ENCODER_ANSWERABILITY_PROVIDER
    )

    assert (
        config.answerability.model
        == DEFAULT_RUNTIME_ANSWERABILITY_MODEL
    )

def test_environment_can_replace_all_provider_and_model_choices():
    config = load_runtime_provider_config(
        {
            EMBEDDING_PROVIDER_ENV: (
                "replacement-embedding-provider"
            ),
            EMBEDDING_MODEL_ENV: (
                "replacement-embedding-model"
            ),
            RERANKER_PROVIDER_ENV: (
                "replacement-reranker-provider"
            ),
            RERANKER_MODEL_ENV: (
                "replacement-reranker-model"
            ),
            PRIMARY_PARSER_PROVIDER_ENV: (
                "replacement-parser-provider"
            ),
            PRIMARY_PARSER_MODEL_ENV: (
                "replacement-primary-parser"
            ),
            FALLBACK_PARSER_PROVIDER_ENV: (
                "replacement-fallback-provider"
            ),
            FALLBACK_PARSER_MODEL_ENV: (
                "replacement-fallback-parser"
            ),
            ANSWERABILITY_PROVIDER_ENV: (
                "replacement-answerability-provider"
            ),
            ANSWERABILITY_MODEL_ENV: (
                "replacement-answerability-model"
            ),
        }
    )

    assert (
        config.retrieval.embedding.provider
        == "replacement-embedding-provider"
    )

    assert (
        config.retrieval.embedding.model
        == "replacement-embedding-model"
    )

    assert (
        config.retrieval.reranker.provider
        == "replacement-reranker-provider"
    )

    assert (
        config.retrieval.reranker.model
        == "replacement-reranker-model"
    )

    assert (
        config.question_parser.primary.provider
        == "replacement-parser-provider"
    )

    assert (
        config.question_parser.primary.model
        == "replacement-primary-parser"
    )

    assert (
        config.question_parser.fallback
        is not None
    )

    assert (
        config.question_parser.fallback.provider
        == "replacement-fallback-provider"
    )

    assert (
        config.question_parser.fallback.model
        == "replacement-fallback-parser"
    )

    assert (
        config.answerability.provider
        == "replacement-answerability-provider"
    )

    assert (
        config.answerability.model
        == "replacement-answerability-model"
    )

def test_environment_can_override_only_one_model():
    config = load_runtime_provider_config(
        {
            EMBEDDING_MODEL_ENV: (
                "replacement-embedding-model"
            )
        }
    )

    assert (
        config.retrieval.embedding.model
        == "replacement-embedding-model"
    )

    assert (
        config.retrieval.embedding.provider
        == SENTENCE_TRANSFORMER_PROVIDER
    )

    assert (
        config.retrieval.reranker.model
        == DEFAULT_RUNTIME_RERANKER_MODEL
    )

def test_rejects_blank_environment_setting():
    with pytest.raises(
        ValueError,
        match=(
            "RAVIN_EMBEDDING_MODEL "
            "cannot be empty."
        ),
    ):
        load_runtime_provider_config(
            {
                EMBEDDING_MODEL_ENV: "   ",
            }
        )
