from pathlib import Path

from rag_agent_platform.graph.base import GraphService
from rag_agent_platform.ingestion.coordinated import CoordinatedIngestionPipeline
from rag_agent_platform.ingestion.pipeline import RealIngestionPipeline
from rag_agent_platform.models import ChildChunk, RetrievedChunk
from rag_agent_platform.storage.file_repository import FileDocumentRepository


class FakeVectorStore:
    def __init__(self) -> None:
        self.chunks: dict[str, ChildChunk] = {}

    def upsert_child_chunks(self, chunks: list[ChildChunk]) -> None:
        self.chunks.update({chunk.chunk_id: chunk for chunk in chunks})

    def delete_document(self, document_id: str) -> None:
        self.chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self.chunks.items()
            if chunk.document_id != document_id
        }

    def count(self, document_id: str | None = None) -> int:
        if document_id is None:
            return len(self.chunks)
        return sum(chunk.document_id == document_id for chunk in self.chunks.values())


class FakeGraphService(GraphService):
    def __init__(self) -> None:
        self.chunk_document_ids: set[str] = set()
        self.deleted_document_ids: list[str] = []

    def build(self, chunks: list[ChildChunk]) -> None:
        self.chunk_document_ids.update(chunk.document_id for chunk in chunks)

    def delete_document(self, document_id: str) -> None:
        self.chunk_document_ids.discard(document_id)
        self.deleted_document_ids.append(document_id)

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        return []


class FakeRefreshableRetriever:
    def __init__(self) -> None:
        self.refresh_count = 0

    def refresh(self) -> None:
        self.refresh_count += 1


def test_coordinated_delete_cascades_to_all_mutable_indexes(tmp_path: Path) -> None:
    repository = FileDocumentRepository(tmp_path / "metadata" / "documents.json")
    vector_store = FakeVectorStore()
    graph_service = FakeGraphService()
    sparse_retriever = FakeRefreshableRetriever()
    pipeline = CoordinatedIngestionPipeline(
        RealIngestionPipeline(repository=repository, vector_store=vector_store),
        repository=repository,
        sparse_retriever=sparse_retriever,
        graph_service=graph_service,
    )
    document_path = tmp_path / "policy.txt"
    document_path.write_text("采购部门负责供应商管理。", encoding="utf-8")

    result = pipeline.ingest(document_path)
    document_id = result.document.document_id
    assert repository.list_parent_chunks([document_id])
    assert repository.list_child_chunks([document_id])
    assert vector_store.count(document_id) > 0
    assert document_id in graph_service.chunk_document_ids

    pipeline.delete_document(document_id)

    assert pipeline.list_documents() == []
    assert repository.list_parent_chunks([document_id]) == []
    assert repository.list_child_chunks([document_id]) == []
    assert vector_store.count(document_id) == 0
    assert graph_service.deleted_document_ids == [document_id]
    assert document_id not in graph_service.chunk_document_ids
    assert sparse_retriever.refresh_count == 2
