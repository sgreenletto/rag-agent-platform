"""Embedding model boundaries and lightweight implementations."""

from rag_agent_platform.embeddings.base import EmbeddingIdentity, EmbeddingModel
from rag_agent_platform.embeddings.factory import build_embedding_model
from rag_agent_platform.embeddings.hash import HashEmbeddingModel
from rag_agent_platform.embeddings.openai_compatible import OpenAICompatibleEmbeddingModel

__all__ = [
    "EmbeddingIdentity",
    "EmbeddingModel",
    "HashEmbeddingModel",
    "OpenAICompatibleEmbeddingModel",
    "build_embedding_model",
]
