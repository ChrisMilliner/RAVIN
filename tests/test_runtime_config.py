import pytest
from backend.core.runtime_config import (
    ProviderModelConfig,
    QuestionParserProviderConfig,
    RetrievalProviderConfig,
    RuntimeProviderConfig,
)

def test_creates_provider_model_config():
    config = ProviderModelConfig(
        provider="replaceable-provider",
        model="replaceable-model",
    )

    assert config.provider == (
        "replaceable-provider"
    )

    assert config.model == (
        "replaceable-model"
    )

def test_rejects_empty_provider():
    with pytest.raises(
        ValueError,
        match="Provider cannot be empty.",
    ):
        ProviderModelConfig(
            provider="   ",
            model="model",
        )

def test_rejects_empty_model():
    with pytest.raises(
        ValueError,
        match="Model cannot be empty.",
    ):
        ProviderModelConfig(
            provider="provider",
            model="   ",
        )

def test_parser_fallback_is_optional():
    primary = ProviderModelConfig(
        provider="parser-provider",
        model="primary-model",
    )

    config = (
        QuestionParserProviderConfig(
            primary=primary,
        )
    )

    assert config.primary is primary
    assert config.fallback is None

def test_runtime_config_preserves_provider_choices():
    embedding = ProviderModelConfig(
        provider="embedding-provider",
        model="embedding-model",
    )

    reranker = ProviderModelConfig(
        provider="reranker-provider",
        model="reranker-model",
    )

    primary_parser = ProviderModelConfig(
        provider="parser-provider",
        model="primary-parser-model",
    )

    fallback_parser = ProviderModelConfig(
        provider="parser-provider",
        model="fallback-parser-model",
    )

    config = RuntimeProviderConfig(
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
    )

    assert (
        config.retrieval.embedding
        is embedding
    )

    assert (
        config.retrieval.reranker
        is reranker
    )

    assert (
        config.question_parser.primary
        is primary_parser
    )

    assert (
        config.question_parser.fallback
        is fallback_parser
    )