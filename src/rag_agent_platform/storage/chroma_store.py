"""ChromaDB-backed vector storage for child chunks."""

from pathlib import Path
from typing import Any

import chromadb

from rag_agent_platform.embeddings import EmbeddingModel, HashEmbeddingModel
from rag_agent_platform.models import ChildChunk


class ChromaVectorStore:
    """Persist child chunk embeddings in ChromaDB."""

    def __init__(
        self,
        persist_directory: str | Path = "data/chroma",
        collection_name: str = "rag_child_chunks",
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self._persist_directory = Path(persist_directory)
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        self._embedding_model = embedding_model or HashEmbeddingModel()
        self._client = chromadb.PersistentClient(path=str(self._persist_directory))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_child_chunks(self, chunks: list[ChildChunk]) -> None:
        """Embed and upsert child chunks."""
        if not chunks:
            return
        texts = [chunk.content for chunk in chunks]
        embeddings = self._embedding_model.embed_texts(texts)
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[self._metadata_for_chroma(chunk) for chunk in chunks],
        )

    def delete_document(self, document_id: str) -> None:
        """Delete all vectors associated with one document."""
        self._collection.delete(where={"document_id": document_id})

    def count(self, document_id: str | None = None) -> int:
        """Return total vector count, optionally for one document."""
        if document_id is None:
            return int(self._collection.count())
        result = self._collection.get(where={"document_id": document_id})
        return len(result["ids"])

    @staticmethod
    def _metadata_for_chroma(chunk: ChildChunk) -> dict[str, Any]:
        metadata = {
            key: ChromaVectorStore._sanitize_metadata_value(value)
            for key, value in chunk.metadata.items()
        }
        metadata.update(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "parent_id": chunk.parent_id,
                "page": chunk.page if chunk.page is not None else -1,
            }
        )
        return metadata

    @staticmethod
    def _sanitize_metadata_value(value: Any) -> str | int | float | bool:
        if isinstance(value, str | int | float | bool):
            return value
        if value is None:
            return ""
        return str(value)
