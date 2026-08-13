from typing import cast
from numpy import ndarray
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

class SentenceTransformerEmbeddingProvider:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self._model = SentenceTransformer(model_name)

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