"""Chroma adapter shaped for the member-two dense retriever."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from rag_agent_platform.models import ChildChunk
from rag_agent_platform.storage.chroma_store import ChromaVectorStore


@dataclass(slots=True)
class DenseSearchHit:
    """Storage-neutral dense search hit with an unnormalized backend score."""

    chunk: ChildChunk
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isfinite(self.score):
            raise ValueError("score must be finite")
        if not self.source.strip():
            raise ValueError("source must not be empty")


class ChromaDenseSearchBackend:
    """Expose Chroma vector search through the member-two backend shape."""

    def __init__(self, vector_store: ChromaVectorStore) -> None:
        self._vector_store = vector_store

    def search(
        self,
        query: str,
        document_ids: list[str] | None,
        limit: int,
    ) -> list[DenseSearchHit]:
        """Return dense search hits for the requested documents."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than 0")
        if document_ids == []:
            return []
        return [
            DenseSearchHit(
                chunk=hit.chunk,
                score=hit.score,
                source=hit.source,
                metadata=hit.metadata,
            )
            for hit in self._vector_store.search(
                query=query,
                document_ids=document_ids,
                limit=limit,
            )
        ]
