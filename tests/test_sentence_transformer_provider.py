import numpy as np
import pytest
import backend.retrieval.sentence_transformer_provider as provider_module
from backend.retrieval.sentence_transformer_provider import (
    SentenceTransformerEmbeddingProvider,
)

TEST_EMBEDDING_MODEL = (
    "example/test-embedding-model"
)

class FakeSentenceTransformer:
    def __init__(
        self,
        model_name: str,
    ) -> None:
        self.model_name = model_name
        self.document_texts = None
        self.query_text = None

    def encode_document(
        self,
        texts,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ):
        self.document_texts = texts

        assert convert_to_numpy is True
        assert show_progress_bar is False

        return np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )

    def encode_query(
        self,
        text: str,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ):
        self.query_text = text

        assert convert_to_numpy is True
        assert show_progress_bar is False

        return np.array(
            [5.0, 6.0]
        )

def install_fake_sentence_transformer(
    monkeypatch,
):
    created_models = []

    def fake_constructor(
        model_name: str,
    ):
        model = FakeSentenceTransformer(
            model_name
        )

        created_models.append(
            model
        )

        return model

    monkeypatch.setattr(
        provider_module,
        "SentenceTransformer",
        fake_constructor,
    )

    return created_models

def test_provider_passes_model_name(
    monkeypatch,
):
    created_models = (
        install_fake_sentence_transformer(
            monkeypatch
        )
    )

    SentenceTransformerEmbeddingProvider(
        model_name=TEST_EMBEDDING_MODEL
    )

    assert len(created_models) == 1

    assert (
        created_models[0].model_name
        == TEST_EMBEDDING_MODEL
    )

def test_provider_embeds_documents(
    monkeypatch,
):
    created_models = (
        install_fake_sentence_transformer(
            monkeypatch
        )
    )

    provider = (
        SentenceTransformerEmbeddingProvider(
            model_name=TEST_EMBEDDING_MODEL
        )
    )

    embeddings = provider.embed_documents(
        (
            "First policy chunk.",
            "Second policy chunk.",
        )
    )

    assert created_models[0].document_texts == [
        "First policy chunk.",
        "Second policy chunk.",
    ]

    assert embeddings == (
        (1.0, 2.0),
        (3.0, 4.0),
    )

def test_provider_embeds_query(
    monkeypatch,
):
    created_models = (
        install_fake_sentence_transformer(
            monkeypatch
        )
    )

    provider = (
        SentenceTransformerEmbeddingProvider(
            model_name=TEST_EMBEDDING_MODEL
        )
    )

    embedding = provider.embed_query(
        "What does the policy say?"
    )

    assert (
        created_models[0].query_text
        == "What does the policy say?"
    )

    assert embedding == (
        5.0,
        6.0,
    )

def test_provider_rejects_empty_model_name(
    monkeypatch,
):
    install_fake_sentence_transformer(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match=(
            "Embedding model name cannot be empty."
        ),
    ):
        SentenceTransformerEmbeddingProvider(
            model_name="   "
        )