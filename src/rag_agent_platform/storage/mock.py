"""In-memory repository used only by the project skeleton."""

from rag_agent_platform.models import ChildChunk, DocumentRecord, ParentChunk
from rag_agent_platform.storage.base import DocumentRepository


class MockDocumentRepository(DocumentRepository):
    """Store documents and chunks in process memory without a database."""

    def __init__(self) -> None:
        self._documents: dict[str, DocumentRecord] = {}
        self._parent_chunks: dict[str, ParentChunk] = {}
        self._child_chunks: dict[str, ChildChunk] = {}

    def save_document(self, document: DocumentRecord) -> None:
        self._documents[document.document_id] = document

    def get_document(self, document_id: str) -> DocumentRecord | None:
        return self._documents.get(document_id)

    def list_documents(self) -> list[DocumentRecord]:
        return sorted(self._documents.values(), key=lambda document: document.filename)

    def delete_document(self, document_id: str) -> None:
        self._documents.pop(document_id, None)
        self._parent_chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self._parent_chunks.items()
            if chunk.document_id != document_id
        }
        self._child_chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self._child_chunks.items()
            if chunk.document_id != document_id
        }

    def save_parent_chunks(self, chunks: list[ParentChunk]) -> None:
        self._parent_chunks.update({chunk.chunk_id: chunk for chunk in chunks})

    def save_child_chunks(self, chunks: list[ChildChunk]) -> None:
        self._child_chunks.update({chunk.chunk_id: chunk for chunk in chunks})
