from math import exp
import pytest
import backend.generation.cross_encoder_entailment as entailment_module
from backend.generation.cross_encoder_entailment import (
    CrossEncoderEntailmentProvider,
)
from backend.generation.entailment import (
    EntailmentPair,
)

class FakeConfig:
    def __init__(
        self,
        id2label,
    ):
        self.id2label = id2label

class FakeUnderlyingModel:
    def __init__(
        self,
        id2label,
    ):
        self.config = FakeConfig(
            id2label
        )

class FakeCrossEncoder:
    def __init__(
        self,
        model_name,
        *,
        id2label=None,
        predictions=None,
    ):
        self.model_name = model_name

        self.model = FakeUnderlyingModel(
            id2label
            or {
                0: "contradiction",
                1: "entailment",
                2: "neutral",
            }
        )

        self.predictions = (
            predictions
            if predictions is not None
            else []
        )

        self.received_pairs = None

    def predict(
        self,
        pairs,
    ):
        self.received_pairs = pairs
        return self.predictions

def test_provider_loads_configured_model(
    monkeypatch,
):
    created = {}

    def fake_factory(
        model_name,
    ):
        model = FakeCrossEncoder(
            model_name,
        )

        created["model"] = model

        return model

    monkeypatch.setattr(
        entailment_module,
        "CrossEncoder",
        fake_factory,
    )

    provider = (
        CrossEncoderEntailmentProvider(
            "test-nli-model"
        )
    )

    assert (
        provider.model_name
        == "test-nli-model"
    )

    assert (
        created["model"].model_name
        == "test-nli-model"
    )

def test_provider_batches_pairs_in_premise_hypothesis_order(
    monkeypatch,
):
    model = FakeCrossEncoder(
        "model",
        predictions=[
            [
                0.0,
                3.0,
                0.0,
            ],
            [
                2.0,
                0.0,
                1.0,
            ],
        ],
    )

    monkeypatch.setattr(
        entailment_module,
        "CrossEncoder",
        lambda model_name: model,
    )

    provider = (
        CrossEncoderEntailmentProvider(
            "model"
        )
    )

    pairs = (
        EntailmentPair(
            premise="Evidence one.",
            hypothesis="Claim one.",
        ),
        EntailmentPair(
            premise="Evidence two.",
            hypothesis="Claim two.",
        ),
    )

    scores = provider.score_entailment(
        pairs
    )

    assert model.received_pairs == [
        (
            "Evidence one.",
            "Claim one.",
        ),
        (
            "Evidence two.",
            "Claim two.",
        ),
    ]

    expected_first = (
        exp(3.0)
        / (
            exp(0.0)
            + exp(3.0)
            + exp(0.0)
        )
    )

    expected_second = (
        exp(0.0)
        / (
            exp(2.0)
            + exp(0.0)
            + exp(1.0)
        )
    )

    assert scores[0] == pytest.approx(
        expected_first
    )

    assert scores[1] == pytest.approx(
        expected_second
    )

def test_provider_discovers_entailment_label_position(
    monkeypatch,
):
    model = FakeCrossEncoder(
        "model",
        id2label={
            0: "NEUTRAL",
            1: "CONTRADICTION",
            2: "ENTAILMENT",
        },
        predictions=[
            [
                0.0,
                0.0,
                4.0,
            ]
        ],
    )

    monkeypatch.setattr(
        entailment_module,
        "CrossEncoder",
        lambda model_name: model,
    )

    provider = (
        CrossEncoderEntailmentProvider(
            "model"
        )
    )

    scores = provider.score_entailment(
        (
            EntailmentPair(
                premise="Evidence.",
                hypothesis="Claim.",
            ),
        )
    )

    assert scores[0] > 0.96

def test_provider_returns_empty_batch_without_prediction(
    monkeypatch,
):
    model = FakeCrossEncoder(
        "model"
    )

    monkeypatch.setattr(
        entailment_module,
        "CrossEncoder",
        lambda model_name: model,
    )

    provider = (
        CrossEncoderEntailmentProvider(
            "model"
        )
    )

    scores = provider.score_entailment(
        ()
    )

    assert scores == ()
    assert model.received_pairs is None

def test_provider_rejects_empty_model_name():
    with pytest.raises(
        ValueError,
        match="model name cannot be empty",
    ):
        CrossEncoderEntailmentProvider(
            "   "
        )

def test_provider_rejects_missing_label_mapping(
    monkeypatch,
):
    model = FakeCrossEncoder(
        "model"
    )

    model.model.config.id2label = None

    monkeypatch.setattr(
        entailment_module,
        "CrossEncoder",
        lambda model_name: model,
    )

    with pytest.raises(
        ValueError,
        match="id2label mapping",
    ):
        CrossEncoderEntailmentProvider(
            "model"
        )

def test_provider_rejects_model_without_entailment_label(
    monkeypatch,
):
    model = FakeCrossEncoder(
        "model",
        id2label={
            0: "contradiction",
            1: "neutral",
        },
    )

    monkeypatch.setattr(
        entailment_module,
        "CrossEncoder",
        lambda model_name: model,
    )

    with pytest.raises(
        ValueError,
        match="does not expose an 'entailment'",
    ):
        CrossEncoderEntailmentProvider(
            "model"
        )

def test_provider_rejects_wrong_prediction_count(
    monkeypatch,
):
    model = FakeCrossEncoder(
        "model",
        predictions=[],
    )

    monkeypatch.setattr(
        entailment_module,
        "CrossEncoder",
        lambda model_name: model,
    )

    provider = (
        CrossEncoderEntailmentProvider(
            "model"
        )
    )

    with pytest.raises(
        ValueError,
        match="unexpected number",
    ):
        provider.score_entailment(
            (
                EntailmentPair(
                    premise="Evidence.",
                    hypothesis="Claim.",
                ),
            )
        )

def test_provider_rejects_prediction_without_entailment_position(
    monkeypatch,
):
    model = FakeCrossEncoder(
        "model",
        id2label={
            0: "contradiction",
            2: "entailment",
        },
        predictions=[
            [
                1.0,
                2.0,
            ]
        ],
    )

    monkeypatch.setattr(
        entailment_module,
        "CrossEncoder",
        lambda model_name: model,
    )

    provider = (
        CrossEncoderEntailmentProvider(
            "model"
        )
    )

    with pytest.raises(
        ValueError,
        match="outside the model prediction row",
    ):
        provider.score_entailment(
            (
                EntailmentPair(
                    premise="Evidence.",
                    hypothesis="Claim.",
                ),
            )
        )