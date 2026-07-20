"""Read-only corpus boundary used by sparse and in-memory retrievers."""

from typing import Protocol, runtime_checkable

from rag_agent_platform.models import ChildChunk


@runtime_checkable
class ChunkCorpus(Protocol):
    """Expose retrievable child chunks without coupling to a storage engine."""

    def list_chunks(self, document_ids: list[str] | None = None) -> list[ChildChunk]:
        """Return a snapshot, optionally restricted to selected documents."""
        ...
