from typing import cast
from numpy import ndarray
from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

class CrossEncoderRerankerProvider:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
    ) -> None:
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