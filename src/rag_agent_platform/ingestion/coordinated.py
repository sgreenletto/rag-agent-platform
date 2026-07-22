"""Keep mutable retrieval and graph indexes synchronized with ingestion."""

from pathlib import Path
from typing import Protocol

from rag_agent_platform.graph.base import GraphService
from rag_agent_platform.ingestion.base import IngestionPipeline
from rag_agent_platform.models import DocumentRecord, IngestionResult
from rag_agent_platform.storage.chunk_corpus import ChildChunkRepository


class RefreshableRetriever(Protocol):
    def refresh(self) -> None:
        """Refresh an index from its backing repository."""
        ...


class CoordinatedIngestionPipeline(IngestionPipeline):
    """Delegate real ingestion and refresh BM25/Graph indexes after mutations."""

    def __init__(
        self,
        pipeline: IngestionPipeline,
        *,
        repository: ChildChunkRepository,
        sparse_retriever: RefreshableRetriever,
        graph_service: GraphService,
    ) -> None:
        self._pipeline = pipeline
        self._repository = repository
        self._sparse_retriever = sparse_retriever
        self._graph_service = graph_service

    def ingest(self, file_path: str | Path) -> IngestionResult:
        result = self._pipeline.ingest(file_path)
        chunks = self._repository.list_child_chunks([result.document.document_id])
        self._sparse_retriever.refresh()
        self._graph_service.build(chunks)
        return result

    def list_documents(self) -> list[DocumentRecord]:
        return self._pipeline.list_documents()

    def delete_document(self, document_id: str) -> None:
        self._pipeline.delete_document(document_id)
        self._graph_service.delete_document(document_id)
        self._sparse_retriever.refresh()
