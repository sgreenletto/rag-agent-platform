from pathlib import Path

import pytest

from rag_agent_platform.embeddings import HashEmbeddingModel
from rag_agent_platform.ingestion.pipeline import RealIngestionPipeline
from rag_agent_platform.storage.chroma_store import ChromaVectorStore
from rag_agent_platform.storage.file_repository import FileDocumentRepository


def build_pipeline(
    tmp_path: Path,
) -> tuple[RealIngestionPipeline, FileDocumentRepository, ChromaVectorStore]:
    repository = FileDocumentRepository(tmp_path / "metadata" / "documents.json")
    vector_store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="pipeline_test",
        embedding_model=HashEmbeddingModel(dimensions=16),
    )
    pipeline = RealIngestionPipeline(repository=repository, vector_store=vector_store)
    return pipeline, repository, vector_store


def test_ingest_saves_document_chunks_and_vectors(tmp_path: Path) -> None:
    file_path = tmp_path / "leave_policy.txt"
    file_path.write_text(
        "Leave policy\nEmployees submit annual leave requests three workdays early.",
        encoding="utf-8",
    )
    pipeline, repository, vector_store = build_pipeline(tmp_path)

    result = pipeline.ingest(file_path)

    assert result.document.filename == "leave_policy.txt"
    assert result.parent_chunk_count >= 1
    assert result.child_chunk_count >= 1
    assert pipeline.list_documents() == [result.document]
    assert repository.list_child_chunks([result.document.document_id])
    assert vector_store.count(result.document.document_id) == result.child_chunk_count


def test_delete_document_cleans_repository_and_vectors(tmp_path: Path) -> None:
    file_path = tmp_path / "leave_policy.txt"
    file_path.write_text(
        "Leave policy\nEmployees submit annual leave requests three workdays early.",
        encoding="utf-8",
    )
    pipeline, repository, vector_store = build_pipeline(tmp_path)
    result = pipeline.ingest(file_path)

    pipeline.delete_document(result.document.document_id)

    assert pipeline.list_documents() == []
    assert repository.list_child_chunks([result.document.document_id]) == []
    assert vector_store.count(result.document.document_id) == 0


def test_ingest_rejects_cleaned_empty_document(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("\x00   \n", encoding="utf-8")
    pipeline, _, _ = build_pipeline(tmp_path)

    with pytest.raises(ValueError, match="empty"):
        pipeline.ingest(file_path)
