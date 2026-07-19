"""Persistence contracts for documents and chunks."""

from abc import ABC, abstractmethod

from rag_agent_platform.models import ChildChunk, DocumentRecord, ParentChunk


class DocumentRepository(ABC):
    """Abstract repository for document metadata and parent/child chunks."""

    @abstractmethod
    def save_document(self, document: DocumentRecord) -> None:
        """Create or replace a document record."""
        raise NotImplementedError

    @abstractmethod
    def get_document(self, document_id: str) -> DocumentRecord | None:
        """Return one document, or None when it does not exist."""
        raise NotImplementedError

    @abstractmethod
    def list_documents(self) -> list[DocumentRecord]:
        """Return all document records."""
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete a document and its stored chunks."""
        raise NotImplementedError

    @abstractmethod
    def save_parent_chunks(self, chunks: list[ParentChunk]) -> None:
        """Persist parent chunks."""
        raise NotImplementedError

    @abstractmethod
    def save_child_chunks(self, chunks: list[ChildChunk]) -> None:
        """Persist child chunks."""
        raise NotImplementedError
