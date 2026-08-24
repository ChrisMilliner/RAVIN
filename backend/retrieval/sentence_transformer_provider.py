from typing import cast
from numpy import ndarray
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddingProvider:
    def __init__(
        self,
        model_name: str,
    ) -> None:
        if not model_name.strip():
            raise ValueError(
                "Embedding model name cannot be empty."
            )

        self._model = SentenceTransformer(
            model_name
        )

    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        embeddings = cast(
            ndarray,
            self._model.encode_document(
                list(texts),
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
        )

        return tuple(
            tuple(float(value) for value in embedding)
            for embedding in embeddings
        )

    def embed_query(
        self,
        text: str,
    ) -> tuple[float, ...]:
        embedding = cast(
            ndarray,
            self._model.encode_query(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
        )

        return tuple(
            float(value)
            for value in embedding
        )