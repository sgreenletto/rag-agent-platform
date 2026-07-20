"""Embedding model boundaries and lightweight implementations."""

from rag_agent_platform.embeddings.base import EmbeddingModel
from rag_agent_platform.embeddings.hash import HashEmbeddingModel

__all__ = ["EmbeddingModel", "HashEmbeddingModel"]
