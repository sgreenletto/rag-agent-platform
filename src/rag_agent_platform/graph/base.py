"""Knowledge graph service contract."""

from abc import ABC, abstractmethod

from rag_agent_platform.models import ChildChunk, RetrievedChunk


class GraphService(ABC):
    """Build and query a graph while exposing normalized retrieval results."""

    @abstractmethod
    def build(self, chunks: list[ChildChunk]) -> None:
        """Build graph data from child chunks."""
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete graph data belonging to one document."""
        raise NotImplementedError

    @abstractmethod
    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Return graph evidence converted to RetrievedChunk."""
        raise NotImplementedError
