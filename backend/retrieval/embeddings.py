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
    """Define the framework-neutral embedding contract used by retrieval.

    Implementations provide separate operations for embedding policy
    documents and user queries so retrieval logic does not depend on a
    specific model library.
    """

    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        """Embed an ordered collection of policy document texts.

        Implementations must return one vector for each supplied text in the
        same order.
        """
        ...

    def embed_query(
        self,
        text: str,
    ) -> tuple[float, ...]:
        """Embed one user query into the provider's retrieval vector space.
        """
        ...