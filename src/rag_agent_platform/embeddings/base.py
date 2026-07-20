"""Embedding model protocol used by vector storage adapters."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingModel(Protocol):
    """Convert texts into dense vectors."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""
        ...
