import pytest
import backend.retrieval.cross_encoder_provider as provider_module
from backend.retrieval.cross_encoder_provider import (
    CrossEncoderRerankerProvider,
)

TEST_RERANKER_MODEL = (
    "example/test-reranker-model"
)

class FakeCrossEncoder:
    def __init__(
        self,
        model_name: str,
    ) -> None:
        self.model_name = model_name
        self.last_pairs = None
        self.last_show_progress_bar = None
        self.last_convert_to_numpy = None

    def predict(
        self,
        pairs,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ):
        self.last_pairs = pairs
        self.last_show_progress_bar = (
            show_progress_bar
        )
        self.last_convert_to_numpy = (
            convert_to_numpy
        )

        return [2.5, -1.25]

def install_fake_cross_encoder(
    monkeypatch,
):
    created_models = []

    def fake_constructor(
        model_name: str,
    ):
        model = FakeCrossEncoder(
            model_name
        )
        created_models.append(model)
        return model

    monkeypatch.setattr(
        provider_module,
        "CrossEncoder",
        fake_constructor,
    )

    return created_models

def test_provider_passes_model_name(
    monkeypatch,
):
    created_models = (
        install_fake_cross_encoder(
            monkeypatch
        )
    )

    CrossEncoderRerankerProvider(
        model_name=TEST_RERANKER_MODEL
    )

    assert (
        created_models[0].model_name
        == TEST_RERANKER_MODEL
    )

def test_provider_scores_query_document_pairs(
    monkeypatch,
):
    created_models = (
        install_fake_cross_encoder(
            monkeypatch
        )
    )

    provider = (
        CrossEncoderRerankerProvider(
            model_name=TEST_RERANKER_MODEL
        )
    )

    scores = provider.score(
        query="What does the policy mean?",
        documents=(
            "First policy chunk.",
            "Second policy chunk.",
        ),
    )

    model = created_models[0]

    assert model.last_pairs == [
        (
            "What does the policy mean?",
            "First policy chunk.",
        ),
        (
            "What does the policy mean?",
            "Second policy chunk.",
        ),
    ]

    assert (
        model.last_show_progress_bar
        is False
    )

    assert (
        model.last_convert_to_numpy
        is True
    )

    assert scores == (
        2.5,
        -1.25,
    )

    assert all(
        isinstance(score, float)
        for score in scores
    )

def test_provider_rejects_empty_query(
    monkeypatch,
):
    install_fake_cross_encoder(
        monkeypatch
    )

    provider = (
        CrossEncoderRerankerProvider(
            model_name=TEST_RERANKER_MODEL
        )
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty.",
    ):
        provider.score(
            query="   ",
            documents=(
                "Policy chunk.",
            ),
        )

def test_provider_rejects_empty_documents(
    monkeypatch,
):
    install_fake_cross_encoder(
        monkeypatch
    )

    provider = (
        CrossEncoderRerankerProvider(
            model_name=TEST_RERANKER_MODEL
        )
    )

    with pytest.raises(
        ValueError,
        match="Documents cannot be empty.",
    ):
        provider.score(
            query="Policy question",
            documents=(),
        )

def test_provider_rejects_empty_model_name(
    monkeypatch,
):
    install_fake_cross_encoder(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match=(
            "Reranker model name cannot be empty."
        ),
    ):
        CrossEncoderRerankerProvider(
            model_name="   "
        )
