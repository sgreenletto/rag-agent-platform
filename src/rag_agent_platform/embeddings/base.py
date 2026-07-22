"""Embedding model protocol used by vector storage adapters."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    """Stable identity used to keep document and query embeddings aligned."""

    provider: str
    model: str
    dimensions: int | None = None
    base_url: str | None = None

    def as_metadata(self) -> dict[str, str | int]:
        """Return Chroma-compatible metadata fields."""
        metadata: dict[str, str | int] = {
            "embedding_provider": self.provider,
            "embedding_model": self.model,
        }
        if self.base_url:
            metadata["embedding_base_url"] = self.base_url
        if self.dimensions is not None:
            metadata["embedding_dimensions"] = self.dimensions
        return metadata


@runtime_checkable
class EmbeddingModel(Protocol):
    """Convert texts into dense vectors."""

    @property
    def identity(self) -> EmbeddingIdentity:
        """Return the model identity used for index compatibility checks."""
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""
        ...
