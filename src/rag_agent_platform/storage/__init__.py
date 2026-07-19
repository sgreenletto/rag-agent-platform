"""Storage contracts and Mock implementation."""

from rag_agent_platform.storage.base import DocumentRepository
from rag_agent_platform.storage.mock import MockDocumentRepository

__all__ = ["DocumentRepository", "MockDocumentRepository"]
