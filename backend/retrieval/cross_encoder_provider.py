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