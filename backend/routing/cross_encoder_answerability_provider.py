"""
Provide the cross-encoder adapter for proposition answerability scoring.

This module implements the neutral AnswerabilityProvider contract using
a configured cross-encoder question-answerability model. It scores
material propositions against retrieved evidence blocks.

The concrete model is isolated behind the provider contract so evidence
assessment logic remains independent of a particular model or library.
"""

from typing import cast
import numpy as np
from numpy import ndarray
from sentence_transformers import CrossEncoder
from backend.routing.answerability import (
    AnswerabilityResult,
)

class CrossEncoderAnswerabilityProvider:
    def __init__(
        self,
        model_name: str,
    ) -> None:
        if not model_name.strip():
            raise ValueError(
                "Answerability model name cannot "
                "be empty."
            )

        self._model = CrossEncoder(
            model_name
        )

    def score(
        self,
        question: str,
        evidence_texts: tuple[
            str,
            ...
        ],
    ) -> AnswerabilityResult:
        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        if not evidence_texts:
            raise ValueError(
                "Answerability evidence cannot "
                "be empty."
            )

        if any(
            not evidence_text.strip()
            for evidence_text
            in evidence_texts
        ):
            raise ValueError(
                "Answerability evidence cannot "
                "contain empty text."
            )

        pairs = [
            (
                question,
                evidence_text,
            )
            for evidence_text
            in evidence_texts
        ]

        raw_scores = cast(
            ndarray,
            self._model.predict(
                pairs,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
        )

        scores = np.asarray(
            raw_scores
        )

        if (
            scores.ndim != 1
            or len(scores) != len(
                evidence_texts
            )
        ):
            raise ValueError(
                "Answerability model must return "
                "one score per evidence text."
            )

        return AnswerabilityResult(
            scores=tuple(
                float(score)
                for score in scores
            )
        )