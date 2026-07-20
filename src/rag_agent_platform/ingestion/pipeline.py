"""Real ingestion pipeline skeleton."""

from hashlib import sha256
from pathlib import Path
from typing import Any

from rag_agent_platform.ingestion.base import IngestionPipeline
from rag_agent_platform.ingestion.chunker import ParentChildChunker
from rag_agent_platform.ingestion.cleaner import TextCleaner
from rag_agent_platform.ingestion.loaders import LoaderRegistry, build_default_loader_registry
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
        self._loader_registry = loader_registry or build_default_loader_registry()
        self._cleaner = cleaner or TextCleaner()
        self._chunker = chunker or ParentChildChunker()
        if repository is None:
            from rag_agent_platform.storage.file_repository import FileDocumentRepository

            repository = FileDocumentRepository()
        self._repository = repository
        self._vector_store = vector_store

    def ingest(self, file_path: str | Path) -> IngestionResult:
        """Ingest one local document."""
        path = Path(file_path)
        loaded = self._loader_registry.load(path)
        content = self._cleaner.clean(loaded.content)
        if not content:
            raise ValueError("document must not be empty after cleaning")

        document_id = self._document_id(path, content)
        filename = str(loaded.metadata.get("filename") or path.name)
        file_type = str(loaded.metadata.get("file_type") or path.suffix.lower().lstrip("."))
        document = DocumentRecord(
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            source_path=str(loaded.metadata.get("source_path") or path.resolve()),
            status="ready",
            metadata=dict(loaded.metadata),
        )
        parent_chunks, child_chunks = self._chunker.split(
            document_id=document_id,
            content=content,
            source=filename,
            file_type=file_type,
        )

        self._repository.save_document(document)
        self._repository.save_parent_chunks(parent_chunks)
        self._repository.save_child_chunks(child_chunks)
        if self._vector_store is not None:
            self._vector_store.upsert_child_chunks(child_chunks)

        return IngestionResult(
            document=document,
            parent_chunk_count=len(parent_chunks),
            child_chunk_count=len(child_chunks),
        )

    def list_documents(self) -> list[DocumentRecord]:
        """Return available document records."""
        return self._repository.list_documents()

    def delete_document(self, document_id: str) -> None:
        """Delete one document and its chunks."""
        self._repository.delete_document(document_id)
        if self._vector_store is not None:
            self._vector_store.delete_document(document_id)

    def count_vectors(self, document_id: str | None = None) -> int:
        """Return indexed vector count for smoke checks and UI status."""
        if self._vector_store is None:
            return 0
        return int(self._vector_store.count(document_id))

    @staticmethod
    def _document_id(path: Path, content: str) -> str:
        digest = sha256()
        digest.update(str(path.resolve()).encode())
        digest.update(b"\0")
        digest.update(content.encode())
        return f"doc-{digest.hexdigest()[:16]}"
