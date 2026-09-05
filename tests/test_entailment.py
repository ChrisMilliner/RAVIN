import pytest
from backend.generation.entailment import (
    EntailmentPair,
    score_entailment,
)

class StubEntailmentProvider:
    def __init__(
        self,
        scores: tuple[float, ...],
    ) -> None:
        self._scores = scores
        self.received_pairs = None

    def score_entailment(
        self,
        pairs: tuple[EntailmentPair, ...],
    ) -> tuple[float, ...]:
        self.received_pairs = pairs
        return self._scores

def test_entailment_pair_rejects_empty_premise():
    with pytest.raises(
        ValueError,
        match="premise cannot be empty",
    ):
        EntailmentPair(
            premise="   ",
            hypothesis="Supported claim.",
        )

def test_entailment_pair_rejects_empty_hypothesis():
    with pytest.raises(
        ValueError,
        match="hypothesis cannot be empty",
    ):
        EntailmentPair(
            premise="Policy evidence.",
            hypothesis="   ",
        )

def test_score_entailment_preserves_batch_order():
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

    provider = StubEntailmentProvider(
        (
            0.91,
            0.24,
        )
    )

    scores = score_entailment(
        provider,
        pairs,
    )

    assert scores == (
        0.91,
        0.24,
    )

    assert provider.received_pairs == pairs

def test_score_entailment_rejects_empty_batch():
    provider = StubEntailmentProvider(
        ()
    )

    with pytest.raises(
        ValueError,
        match="At least one entailment pair",
    ):
        score_entailment(
            provider,
            (),
        )

def test_score_entailment_rejects_wrong_score_count():
    pairs = (
        EntailmentPair(
            premise="Evidence.",
            hypothesis="Claim.",
        ),
    )

    provider = StubEntailmentProvider(
        (
            0.91,
            0.92,
        )
    )

    with pytest.raises(
        ValueError,
        match="unexpected number of scores",
    ):
        score_entailment(
            provider,
            pairs,
        )

@pytest.mark.parametrize(
    "invalid_score",
    (
        -0.01,
        1.01,
        float("inf"),
        float("-inf"),
        float("nan"),
    ),
)
def test_score_entailment_rejects_invalid_probability(
    invalid_score,
):
    pairs = (
        EntailmentPair(
            premise="Evidence.",
            hypothesis="Claim.",
        ),
    )

    provider = StubEntailmentProvider(
        (
            invalid_score,
        )
    )

    with pytest.raises(
        ValueError,
        match="finite probabilities",
    ):
        score_entailment(
            provider,
            pairs,
        )