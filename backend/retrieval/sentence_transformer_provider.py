"""
Provide the Sentence Transformer adapter for semantic embeddings.

This module implements the framework-neutral EmbeddingProvider contract
using a configured Sentence Transformer model for policy documents and
user questions.

The adapter owns library-specific model interaction while semantic
indexing and retrieval remain independent of the concrete embedding
implementation.
"""

from typing import cast
from numpy import ndarray
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddingProvider:
    """Implement RAVIN embeddings with a Sentence Transformer model.

    The concrete model is loaded when the adapter is constructed and remains
    behind the neutral EmbeddingProvider contract used by semantic indexing.
    """

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
        """Embed policy documents using the model's document encoding operation.

        The returned immutable vectors preserve the input document order.
        """
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
        """Embed a user query using the model's query encoding operation.

        The resulting immutable vector can be compared with indexed document
        embeddings by the semantic retrieval layer.
        """
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