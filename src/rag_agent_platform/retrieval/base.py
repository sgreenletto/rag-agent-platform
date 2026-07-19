"""Retriever contract shared by all retrieval strategies."""

from abc import ABC, abstractmethod

from rag_agent_platform.models import RetrievedChunk


class BaseRetriever(ABC):
    """Return normalized chunks ordered from most to least relevant."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Retrieve evidence as RetrievedChunk values only."""
        raise NotImplementedError
