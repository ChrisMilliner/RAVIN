import numpy as np
import pytest
import backend.routing.cross_encoder_answerability_provider as provider_module
from backend.routing.cross_encoder_answerability_provider import (
    CrossEncoderAnswerabilityProvider,
)

TEST_ANSWERABILITY_MODEL = (
    "example/test-answerability-model"
)

class FakeCrossEncoder:
    def __init__(
        self,
        model_name: str,
        scores: np.ndarray,
    ) -> None:
        self.model_name = model_name
        self.scores = scores
        self.received_pairs = None

    def predict(
        self,
        pairs,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ):
        self.received_pairs = pairs

        assert show_progress_bar is False
        assert convert_to_numpy is True

        return self.scores

def install_fake_cross_encoder(
    monkeypatch,
    scores: np.ndarray,
):
    created_models = []

    def fake_constructor(
        model_name: str,
    ):
        model = FakeCrossEncoder(
            model_name,
            scores,
        )

        created_models.append(
            model
        )

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
            monkeypatch,
            np.array(
                [0.5]
            ),
        )
    )

    CrossEncoderAnswerabilityProvider(
        model_name=TEST_ANSWERABILITY_MODEL
    )

    assert len(created_models) == 1

    assert (
        created_models[0].model_name
        == TEST_ANSWERABILITY_MODEL
    )

def test_provider_scores_question_evidence_pairs(
    monkeypatch,
):
    created_models = (
        install_fake_cross_encoder(
            monkeypatch,
            np.array(
                [
                    0.95,
                    0.10,
                ]
            ),
        )
    )

    provider = (
        CrossEncoderAnswerabilityProvider(
            model_name=(
                TEST_ANSWERABILITY_MODEL
            )
        )
    )

    result = provider.score(
        "What does the policy require?",
        (
            "Relevant policy evidence.",
            "Unrelated policy evidence.",
        ),
    )

    assert created_models[0].received_pairs == [
        (
            "What does the policy require?",
            "Relevant policy evidence.",
        ),
        (
            "What does the policy require?",
            "Unrelated policy evidence.",
        ),
    ]

    assert result.scores == (
        pytest.approx(0.95),
        pytest.approx(0.10),
    )

def test_provider_rejects_empty_model_name(
    monkeypatch,
):
    install_fake_cross_encoder(
        monkeypatch,
        np.array(
            [0.5]
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Answerability model name cannot "
            "be empty."
        ),
    ):
        CrossEncoderAnswerabilityProvider(
            model_name="   "
        )

def test_provider_rejects_unexpected_score_shape(
    monkeypatch,
):
    install_fake_cross_encoder(
        monkeypatch,
        np.array(
            [
                [
                    0.9,
                    0.1,
                ]
            ]
        ),
    )

    provider = (
        CrossEncoderAnswerabilityProvider(
            model_name=(
                TEST_ANSWERABILITY_MODEL
            )
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Answerability model must return "
            "one score per evidence text."
        ),
    ):
        provider.score(
            "Question?",
            (
                "Evidence.",
            ),
        )

def test_provider_does_not_clamp_invalid_scores(
    monkeypatch,
):
    install_fake_cross_encoder(
        monkeypatch,
        np.array(
            [1.20]
        ),
    )

    provider = (
        CrossEncoderAnswerabilityProvider(
            model_name=(
                TEST_ANSWERABILITY_MODEL
            )
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Answerability scores must be "
            "between 0 and 1."
        ),
    ):
        provider.score(
            "Question?",
            (
                "Evidence.",
            ),
        )