"""
Sentence Transformers CrossEncoder entailment provider.

This adapter implements RAVIN's model-neutral EntailmentProvider
contract using a Natural Language Inference (NLI) CrossEncoder.

The NLI model receives:

    premise
        = approved policy evidence

    hypothesis
        = generated factual claim

A typical NLI model produces scores for:

    contradiction
    entailment
    neutral

RAVIN only needs the entailment probability for claim-grounding
validation. The other labels remain internal to this adapter.

The entailment label position is discovered from the model
configuration rather than hard-coded so that the business logic
does not depend on one specific model's label ordering.
"""

from math import exp
from typing import Any
from sentence_transformers import CrossEncoder
from backend.generation.entailment import (
    EntailmentPair,
)

class CrossEncoderEntailmentProvider:
    """NLI CrossEncoder implementation of entailment scoring."""

    def __init__(
        self,
        model_name: str,
    ) -> None:
        if not model_name.strip():
            raise ValueError(
                "Entailment model name cannot be empty."
            )

        self._model_name = model_name
        self._model = CrossEncoder(
            model_name
        )

        self._entailment_index = (
            _resolve_entailment_index(
                self._model
            )
        )

    @property
    def model_name(self) -> str:
        """Return the configured model name."""

        return self._model_name

    def score_entailment(
        self,
        pairs: tuple[
            EntailmentPair,
            ...,
        ],
    ) -> tuple[float, ...]:
        """
        Return one entailment probability for each pair.

        All supplied pairs are sent to the CrossEncoder together so
        that claim validation can use efficient batch inference.
        """

        if not pairs:
            return ()

        model_pairs = [
            (
                pair.premise,
                pair.hypothesis,
            )
            for pair in pairs
        ]

        raw_scores = self._model.predict(
            model_pairs
        )

        if len(raw_scores) != len(pairs):
            raise ValueError(
                "Entailment model returned an unexpected "
                "number of prediction rows."
            )

        return tuple(
            _extract_entailment_probability(
                row,
                self._entailment_index,
            )
            for row in raw_scores
        )

def _resolve_entailment_index(
    model: Any,
) -> int:
    """
    Find the model output position representing entailment.

    The provider requires a labelled NLI model. Models that expose
    only generic labels cannot safely be used because RAVIN would
    not know which output represents entailment.
    """

    model_object = getattr(
        model,
        "model",
        None,
    )

    config = getattr(
        model_object,
        "config",
        None,
    )

    id_to_label = getattr(
        config,
        "id2label",
        None,
    )

    if not isinstance(
        id_to_label,
        dict,
    ):
        raise ValueError(
            "Entailment model must expose an id2label mapping."
        )

    for raw_index, raw_label in (
        id_to_label.items()
    ):
        label = str(
            raw_label
        ).strip().lower()

        if label == "entailment":
            try:
                return int(
                    raw_index
                )
            except (
                TypeError,
                ValueError,
            ) as error:
                raise ValueError(
                    "Entailment label index must be numeric."
                ) from error

    raise ValueError(
        "Entailment model does not expose an "
        "'entailment' output label."
    )

def _extract_entailment_probability(
    raw_scores: Any,
    entailment_index: int,
) -> float:
    """Convert one model prediction row into entailment probability."""

    try:
        scores = tuple(
            float(score)
            for score in raw_scores
        )
    except TypeError as error:
        raise ValueError(
            "Entailment model prediction row must "
            "contain multiple label scores."
        ) from error

    if entailment_index >= len(scores):
        raise ValueError(
            "Entailment label index is outside the "
            "model prediction row."
        )

    probabilities = _softmax(
        scores
    )

    return probabilities[
        entailment_index
    ]

def _softmax(
    values: tuple[float, ...],
) -> tuple[float, ...]:
    """Convert model logits into normalized probabilities."""

    if not values:
        raise ValueError(
            "Entailment model returned an empty "
            "prediction row."
        )

    maximum = max(
        values
    )

    exponentials = tuple(
        exp(
            value - maximum
        )
        for value in values
    )

    total = sum(
        exponentials
    )

    return tuple(
        value / total
        for value in exponentials
    )