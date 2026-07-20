from pathlib import Path

from rag_agent_platform.embeddings import HashEmbeddingModel
from rag_agent_platform.ingestion.pipeline import RealIngestionPipeline
from rag_agent_platform.storage import ChromaDenseSearchBackend, RepositoryChunkCorpus
from rag_agent_platform.storage.chroma_store import ChromaVectorStore
from rag_agent_platform.storage.file_repository import FileDocumentRepository


def test_member1_pipeline_feeds_member2_retrieval_contract(tmp_path: Path) -> None:
    repository = FileDocumentRepository(tmp_path / "metadata" / "documents.json")
    vector_store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="member1_integration_ready",
        embedding_model=HashEmbeddingModel(dimensions=16),
    )
    pipeline = RealIngestionPipeline(repository=repository, vector_store=vector_store)

    results = [
        pipeline.ingest(Path("tests/fixtures/leave_policy.txt")),
        pipeline.ingest(Path("tests/fixtures/sample.pdf")),
        pipeline.ingest(Path("tests/fixtures/sample.docx")),
    ]
    document_ids = [result.document.document_id for result in results]
    child_chunk_count = sum(result.child_chunk_count for result in results)

    corpus = RepositoryChunkCorpus(repository)
    dense_backend = ChromaDenseSearchBackend(vector_store)
    hits = dense_backend.search("policy approval", document_ids=document_ids, limit=5)

    assert len(pipeline.list_documents()) == 3
    assert len(corpus.list_chunks(document_ids)) == child_chunk_count
    assert vector_store.count() == child_chunk_count
    assert hits
    assert {hit.chunk.document_id for hit in hits}.issubset(set(document_ids))
    assert all(hit.source for hit in hits)

    deleted_document_id = results[0].document.document_id
    pipeline.delete_document(deleted_document_id)

    assert repository.get_document(deleted_document_id) is None
    assert corpus.list_chunks([deleted_document_id]) == []
    assert vector_store.count(deleted_document_id) == 0
