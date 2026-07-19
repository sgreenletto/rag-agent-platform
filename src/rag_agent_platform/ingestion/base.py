"""Document ingestion contracts."""

from abc import ABC, abstractmethod
from pathlib import Path

from rag_agent_platform.models import DocumentRecord, IngestionResult


class IngestionPipeline(ABC):
    """Load, split and persist supported documents."""

    @abstractmethod
    def ingest(self, file_path: str | Path) -> IngestionResult:
        """Ingest one local document."""
        raise NotImplementedError

    @abstractmethod
    def list_documents(self) -> list[DocumentRecord]:
        """Return available document records."""
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete one document and its chunks."""
        raise NotImplementedError
