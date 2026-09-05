from typing import cast
import pytest
from backend.routing.answerability import (
    AnswerabilityProvider,
    AnswerabilityResult,
    score_answerability,
)

class FakeAnswerabilityProvider:
    def __init__(
        self,
        result: AnswerabilityResult,
    ) -> None:
        self.result = result
        self.received_question: (
            str | None
        ) = None
        self.received_evidence: (
            tuple[str, ...] | None
        ) = None

    def score(
        self,
        question: str,
        evidence_texts: tuple[
            str,
            ...
        ],
    ) -> AnswerabilityResult:
        self.received_question = question
        self.received_evidence = (
            evidence_texts
        )

        return self.result

def test_result_preserves_scores():
    result = AnswerabilityResult(
        scores=(
            0.25,
            0.90,
            0.60,
        )
    )

    assert result.scores == (
        0.25,
        0.90,
        0.60,
    )

def test_result_reports_strongest_score():
    result = AnswerabilityResult(
        scores=(
            0.25,
            0.90,
            0.60,
        )
    )

    assert result.strongest_score == (
        0.90
    )

def test_result_rejects_empty_scores():
    with pytest.raises(
        ValueError,
        match=(
            "Answerability scores cannot "
            "be empty."
        ),
    ):
        AnswerabilityResult(
            scores=()
        )

@pytest.mark.parametrize(
    "score",
    (
        -0.01,
        1.01,
    ),
)
def test_result_rejects_score_outside_range(
    score: float,
):
    with pytest.raises(
        ValueError,
        match=(
            "Answerability scores must be "
            "between 0 and 1."
        ),
    ):
        AnswerabilityResult(
            scores=(
                score,
            )
        )

def test_scores_question_against_evidence():
    expected = AnswerabilityResult(
        scores=(
            0.95,
            0.20,
        )
    )

    provider = (
        FakeAnswerabilityProvider(
            expected
        )
    )

    evidence = (
        "Relevant policy evidence.",
        "Unrelated policy evidence.",
    )

    result = score_answerability(
        "  What does the policy require?  ",
        evidence,
        provider,
    )

    assert result is expected

    assert provider.received_question == (
        "What does the policy require?"
    )

    assert provider.received_evidence == (
        evidence
    )

def test_rejects_empty_question():
    provider = (
        FakeAnswerabilityProvider(
            AnswerabilityResult(
                scores=(
                    0.5,
                )
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="Question cannot be empty.",
    ):
        score_answerability(
            "   ",
            (
                "Policy evidence.",
            ),
            provider,
        )

def test_rejects_empty_evidence():
    provider = (
        FakeAnswerabilityProvider(
            AnswerabilityResult(
                scores=(
                    0.5,
                )
            )
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Answerability evidence cannot "
            "be empty."
        ),
    ):
        score_answerability(
            "Question?",
            (),
            provider,
        )

def test_rejects_blank_evidence_text():
    provider = (
        FakeAnswerabilityProvider(
            AnswerabilityResult(
                scores=(
                    0.5,
                )
            )
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Answerability evidence cannot "
            "contain empty text."
        ),
    ):
        score_answerability(
            "Question?",
            (
                "Valid evidence.",
                "   ",
            ),
            provider,
        )

def test_rejects_invalid_provider_result():
    class InvalidProvider:
        def score(
            self,
            question: str,
            evidence_texts: tuple[
                str,
                ...
            ],
        ) -> AnswerabilityResult:
            return cast(
                AnswerabilityResult,
                "invalid",
            )

    with pytest.raises(
        ValueError,
        match=(
            "Answerability provider must return "
            "an AnswerabilityResult."
        ),
    ):
        score_answerability(
            "Question?",
            (
                "Evidence.",
            ),
            InvalidProvider(),
        )

def test_rejects_wrong_score_count():
    provider = (
        FakeAnswerabilityProvider(
            AnswerabilityResult(
                scores=(
                    0.9,
                )
            )
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Answerability provider must return "
            "one score per evidence text."
        ),
    ):
        score_answerability(
            "Question?",
            (
                "First evidence.",
                "Second evidence.",
            ),
            provider,
        )