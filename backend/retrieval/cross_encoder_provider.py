"""
Provide the cross-encoder adapter used for retrieval reranking.

This module implements the framework-neutral reranker contract with a
configured cross-encoder model. It scores query and candidate-text
pairs so initially retrieved policy chunks can be reordered by
relevance.

The concrete model remains isolated behind the reranker provider
interface and can be replaced through runtime composition.
"""

from typing import cast
from numpy import ndarray
from sentence_transformers import CrossEncoder

class CrossEncoderRerankerProvider:
    def __init__(
        self,
        model_name: str,
    ) -> None:
        if not model_name.strip():
            raise ValueError(
                "Reranker model name cannot be empty."
            )

        self._model = CrossEncoder(
            model_name
        )

    def score(
        self,
        query: str,
        documents: tuple[str, ...],
    ) -> tuple[float, ...]:
        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not documents:
            raise ValueError(
                "Documents cannot be empty."
            )

        pairs = [
            (
                query,
                document,
            )
            for document in documents
        ]

        scores = cast(
            ndarray,
            self._model.predict(
                pairs,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
        )

        return tuple(
            float(score)
            for score in scores
        )