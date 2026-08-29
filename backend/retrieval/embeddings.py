"""
Define the framework-neutral embedding provider contract.

Embedding providers convert policy text and user questions into vector
representations used by semantic retrieval. The contract separates
semantic-index logic from the specific embedding model or library used
at runtime.

This boundary allows embedding models to be replaced without changing
the retrieval algorithms that consume their vectors.
"""

from typing import Protocol

class EmbeddingProvider(Protocol):
    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        ...

    def embed_query(
        self,
        text: str,
    ) -> tuple[float, ...]:
        ...