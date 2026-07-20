"""JSON-backed document repository for the ingestion MVP."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rag_agent_platform.models import ChildChunk, DocumentRecord, ParentChunk
from rag_agent_platform.storage.base import DocumentRepository


class FileDocumentRepository(DocumentRepository):
    """Persist documents and chunks in a small JSON file."""

    def __init__(self, storage_path: str | Path = "data/metadata/documents.json") -> None:
        self._storage_path = Path(storage_path)
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._documents: dict[str, DocumentRecord] = {}
        self._parent_chunks: dict[str, ParentChunk] = {}
        self._child_chunks: dict[str, ChildChunk] = {}
        self._load()

    def save_document(self, document: DocumentRecord) -> None:
        self._documents[document.document_id] = document
        self._flush()

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
        self._flush()

    def save_parent_chunks(self, chunks: list[ParentChunk]) -> None:
        self._parent_chunks.update({chunk.chunk_id: chunk for chunk in chunks})
        self._flush()

    def save_child_chunks(self, chunks: list[ChildChunk]) -> None:
        self._child_chunks.update({chunk.chunk_id: chunk for chunk in chunks})
        self._flush()

    def get_parent_chunk(self, chunk_id: str) -> ParentChunk | None:
        """Return one parent chunk for parent-context fallback."""
        return self._parent_chunks.get(chunk_id)

    def list_parent_chunks(self, document_ids: list[str] | None = None) -> list[ParentChunk]:
        """Return parent chunks, optionally filtered by document IDs."""
        allowed_ids = None if document_ids is None else set(document_ids)
        return [
            chunk
            for chunk in self._parent_chunks.values()
            if allowed_ids is None or chunk.document_id in allowed_ids
        ]

    def list_child_chunks(self, document_ids: list[str] | None = None) -> list[ChildChunk]:
        """Return child chunks, optionally filtered by document IDs."""
        allowed_ids = None if document_ids is None else set(document_ids)
        return [
            chunk
            for chunk in self._child_chunks.values()
            if allowed_ids is None or chunk.document_id in allowed_ids
        ]

    def _load(self) -> None:
        if not self._storage_path.exists():
            return
        raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
        self._documents = {
            document_id: DocumentRecord(**document)
            for document_id, document in raw.get("documents", {}).items()
        }
        self._parent_chunks = {
            chunk_id: ParentChunk(**chunk)
            for chunk_id, chunk in raw.get("parent_chunks", {}).items()
        }
        self._child_chunks = {
            chunk_id: ChildChunk(**chunk) for chunk_id, chunk in raw.get("child_chunks", {}).items()
        }

    def _flush(self) -> None:
        payload: dict[str, Any] = {
            "documents": {
                document_id: asdict(document) for document_id, document in self._documents.items()
            },
            "parent_chunks": {
                chunk_id: asdict(chunk) for chunk_id, chunk in self._parent_chunks.items()
            },
            "child_chunks": {
                chunk_id: asdict(chunk) for chunk_id, chunk in self._child_chunks.items()
            },
        }
        self._storage_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
