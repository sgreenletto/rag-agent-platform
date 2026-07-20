"""Real ingestion pipeline skeleton."""

from pathlib import Path
from typing import Any

from rag_agent_platform.ingestion.base import IngestionPipeline
from rag_agent_platform.ingestion.chunker import ParentChildChunker
from rag_agent_platform.ingestion.cleaner import TextCleaner
from rag_agent_platform.ingestion.loaders import LoaderRegistry
from rag_agent_platform.models import DocumentRecord, IngestionResult
from rag_agent_platform.storage.base import DocumentRepository


class RealIngestionPipeline(IngestionPipeline):
    """Coordinate loading, cleaning, chunking, metadata storage and indexing."""

    def __init__(
        self,
        *,
        loader_registry: LoaderRegistry | None = None,
        cleaner: TextCleaner | None = None,
        chunker: ParentChildChunker | None = None,
        repository: DocumentRepository | None = None,
        vector_store: Any | None = None,
    ) -> None:
        self._loader_registry = loader_registry or LoaderRegistry()
        self._cleaner = cleaner or TextCleaner()
        self._chunker = chunker or ParentChildChunker()
        self._repository = repository
        self._vector_store = vector_store

    def ingest(self, file_path: str | Path) -> IngestionResult:
        """Ingest one local document.

        The skeleton intentionally stops before real loading and persistence.
        Step-specific implementations will fill this method while preserving
        the public IngestionPipeline contract.
        """
        raise NotImplementedError("real ingestion is not implemented yet")

    def list_documents(self) -> list[DocumentRecord]:
        """Return available document records."""
        if self._repository is None:
            return []
        return self._repository.list_documents()

    def delete_document(self, document_id: str) -> None:
        """Delete one document and its chunks."""
        if self._repository is None:
            raise RuntimeError("document repository is not configured")
        self._repository.delete_document(document_id)
