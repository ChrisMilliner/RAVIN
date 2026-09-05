"""
Neutral entailment contracts for grounded answer validation.

Generated answer claims need a different validation task from
question answerability.

For entailment:

    premise
        = approved policy evidence

    hypothesis
        = factual claim generated from that evidence

An EntailmentProvider returns the probability that the premise
supports, or entails, the hypothesis.

This module intentionally contains no model-specific code. Concrete
providers may use NLI cross-encoders or other implementations without
changing RAVIN's claim-grounding business logic.

Batch scoring is part of the contract because one generated claim may
need to be compared with several candidate evidence windows.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

@dataclass(frozen=True)
class EntailmentPair:
    """One evidence premise and generated-claim hypothesis."""

    premise: str
    hypothesis: str

    def __post_init__(self) -> None:
        if not self.premise.strip():
            raise ValueError(
                "Entailment premise cannot be empty."
            )

        if not self.hypothesis.strip():
            raise ValueError(
                "Entailment hypothesis cannot be empty."
            )

class EntailmentProvider(Protocol):
    """Model-neutral provider for batched entailment scoring."""

    def score_entailment(
        self,
        pairs: tuple[EntailmentPair, ...],
    ) -> tuple[float, ...]:
        """Return one entailment probability for each supplied pair."""
        ...

def score_entailment(
    provider: EntailmentProvider,
    pairs: tuple[EntailmentPair, ...],
) -> tuple[float, ...]:
    """
    Score and validate a batch of premise-hypothesis pairs.

    The provider must return exactly one finite probability between
    zero and one for every supplied pair.
    """

    if not pairs:
        raise ValueError(
            "At least one entailment pair is required."
        )

    scores = tuple(
        provider.score_entailment(
            pairs
        )
    )

    if len(scores) != len(pairs):
        raise ValueError(
            "Entailment provider returned an unexpected "
            "number of scores."
        )

    for score in scores:
        if (
            not isfinite(score)
            or score < 0.0
            or score > 1.0
        ):
            raise ValueError(
                "Entailment scores must be finite "
                "probabilities between 0 and 1."
            )

    return scores