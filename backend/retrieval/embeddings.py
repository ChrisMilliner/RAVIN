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