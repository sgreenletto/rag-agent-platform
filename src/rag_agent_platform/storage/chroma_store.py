"""ChromaDB-backed vector storage for child chunks."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb

from rag_agent_platform.embeddings import EmbeddingIdentity, EmbeddingModel, HashEmbeddingModel
from rag_agent_platform.models import ChildChunk


@dataclass(slots=True)
class ChromaVectorHit:
    """Raw Chroma search hit converted to project chunk data."""

    chunk: ChildChunk
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


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
        collection_metadata = self._collection_metadata(self._embedding_model.identity)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata=collection_metadata,
        )
        self._ensure_embedding_compatible(collection_metadata)

    def upsert_child_chunks(self, chunks: list[ChildChunk]) -> None:
        """Embed and upsert child chunks."""
        if not chunks:
            return
        texts = [chunk.content for chunk in chunks]
        embeddings = self._embedding_model.embed_texts(texts)
        self._ensure_embedding_compatible(self._collection_metadata(self._embedding_model.identity))
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

    def search(
        self,
        query: str,
        document_ids: list[str] | None = None,
        limit: int = 5,
    ) -> list[ChromaVectorHit]:
        """Search child chunks by query text."""
        if limit <= 0:
            raise ValueError("limit must be greater than 0")
        if document_ids == []:
            return []
        query_embedding = self._embedding_model.embed_texts([query])[0]
        self._ensure_embedding_compatible(self._collection_metadata(self._embedding_model.identity))
        raw_result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=self._where_for_document_ids(document_ids),
            include=["documents", "metadatas", "distances"],
        )
        ids = raw_result["ids"][0]
        documents = raw_result["documents"][0]
        metadatas = raw_result["metadatas"][0]
        distances = raw_result["distances"][0]
        return [
            self._to_hit(
                chunk_id=chunk_id,
                content=content,
                metadata=dict(metadata or {}),
                distance=float(distance),
            )
            for chunk_id, content, metadata, distance in zip(
                ids,
                documents,
                metadatas,
                distances,
                strict=True,
            )
        ]

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
    def _where_for_document_ids(document_ids: list[str] | None) -> dict[str, Any] | None:
        if document_ids is None:
            return None
        if len(document_ids) == 1:
            return {"document_id": document_ids[0]}
        return {"document_id": {"$in": document_ids}}

    @staticmethod
    def _to_hit(
        *,
        chunk_id: str,
        content: str,
        metadata: dict[str, Any],
        distance: float,
    ) -> ChromaVectorHit:
        document_id = str(metadata["document_id"])
        parent_id = str(metadata["parent_id"])
        page = metadata.get("page")
        normalized_page = None if page in (None, "", -1) else int(page)
        source = str(
            metadata.get("source")
            or metadata.get("filename")
            or metadata.get("source_path")
            or document_id
        )
        chunk = ChildChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            parent_id=parent_id,
            content=content,
            page=normalized_page,
            metadata=metadata,
        )
        return ChromaVectorHit(
            chunk=chunk,
            score=distance,
            source=source,
            metadata={"chroma_distance": distance},
        )

    @staticmethod
    def _sanitize_metadata_value(value: Any) -> str | int | float | bool:
        if isinstance(value, str | int | float | bool):
            return value
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _collection_metadata(identity: EmbeddingIdentity) -> dict[str, str | int]:
        metadata = {"hnsw:space": "cosine"}
        metadata.update(identity.as_metadata())
        return metadata

    def _ensure_embedding_compatible(self, expected: dict[str, str | int]) -> None:
        actual = self._collection.metadata or {}
        mismatches = {
            key: (actual.get(key), expected_value)
            for key, expected_value in expected.items()
            if key != "hnsw:space"
            and actual.get(key) is not None
            and actual.get(key) != expected_value
        }
        if mismatches:
            details = ", ".join(
                f"{key}: existing={existing!r}, configured={configured!r}"
                for key, (existing, configured) in mismatches.items()
            )
            raise RuntimeError(
                "Chroma collection embedding configuration does not match current settings. "
                f"{details}. Delete the old Chroma data and re-ingest documents."
            )
        missing_metadata = {
            key: value
            for key, value in expected.items()
            if key != "hnsw:space" and actual.get(key) is None
        }
        if missing_metadata:
            required_identity_keys = {"embedding_provider", "embedding_model"}
            missing_required_identity = required_identity_keys.intersection(missing_metadata)
            if missing_required_identity and int(self._collection.count()) > 0:
                missing = ", ".join(sorted(missing_required_identity))
                raise RuntimeError(
                    "Chroma collection has existing vectors but no embedding configuration "
                    f"metadata ({missing}). Delete the old Chroma data and re-ingest documents."
                )
            updated = {key: value for key, value in actual.items() if key != "hnsw:space"}
            updated.update(missing_metadata)
            self._collection.modify(metadata=updated)
