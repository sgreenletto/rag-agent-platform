"""Repository-backed chunk corpus for sparse retrievers."""

from typing import Protocol, runtime_checkable

from rag_agent_platform.models import ChildChunk


@runtime_checkable
class ChildChunkRepository(Protocol):
    """Repository capability required by RepositoryChunkCorpus."""

    def list_child_chunks(self, document_ids: list[str] | None = None) -> list[ChildChunk]:
        """Return child chunks, optionally filtered by document IDs."""
        ...


class RepositoryChunkCorpus:
    """Expose repository child chunks through the member-two corpus shape."""

    def __init__(self, repository: ChildChunkRepository) -> None:
        self._repository = repository

    def list_chunks(self, document_ids: list[str] | None = None) -> list[ChildChunk]:
        """Return retrievable child chunks for BM25 and in-memory retrievers."""
        if document_ids == []:
            return []
        return self._repository.list_child_chunks(document_ids)
