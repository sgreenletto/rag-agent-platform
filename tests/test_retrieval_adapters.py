from pathlib import Path

import pytest

from rag_agent_platform.embeddings import HashEmbeddingModel
from rag_agent_platform.models import ChildChunk
from rag_agent_platform.storage import ChromaDenseSearchBackend, RepositoryChunkCorpus
from rag_agent_platform.storage.chroma_store import ChromaVectorStore
from rag_agent_platform.storage.file_repository import FileDocumentRepository


def build_child(
    chunk_id: str,
    document_id: str,
    parent_id: str,
    content: str,
) -> ChildChunk:
    return ChildChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        parent_id=parent_id,
        content=content,
        metadata={
            "source": "leave_policy.txt",
            "file_type": "txt",
            "page": None,
        },
    )


def test_repository_chunk_corpus_lists_all_and_filtered_chunks(tmp_path: Path) -> None:
    repository = FileDocumentRepository(tmp_path / "documents.json")
    repository.save_child_chunks(
        [
            build_child("child-1", "doc-1", "parent-1", "员工请假制度"),
            build_child("child-2", "doc-2", "parent-2", "采购申请流程"),
        ]
    )
    corpus = RepositoryChunkCorpus(repository)

    assert len(corpus.list_chunks()) == 2
    assert [chunk.document_id for chunk in corpus.list_chunks(["doc-2"])] == ["doc-2"]
    assert corpus.list_chunks([]) == []


def test_chroma_dense_backend_returns_member_two_hit_shape(tmp_path: Path) -> None:
    vector_store = ChromaVectorStore(
        persist_directory=tmp_path,
        collection_name="adapter_test",
        embedding_model=HashEmbeddingModel(dimensions=16),
    )
    child = build_child("child-1", "doc-1", "parent-1", "员工申请年假需要提前提交")
    vector_store.upsert_child_chunks([child])
    backend = ChromaDenseSearchBackend(vector_store)

    hits = backend.search("年假申请", document_ids=["doc-1"], limit=1)

    assert len(hits) == 1
    assert hits[0].chunk.chunk_id == "child-1"
    assert hits[0].chunk.document_id == "doc-1"
    assert hits[0].chunk.parent_id == "parent-1"
    assert hits[0].source == "leave_policy.txt"
    assert "chroma_distance" in hits[0].metadata


def test_chroma_dense_backend_respects_empty_document_filter(tmp_path: Path) -> None:
    backend = ChromaDenseSearchBackend(
        ChromaVectorStore(
            persist_directory=tmp_path,
            collection_name="empty_filter_test",
            embedding_model=HashEmbeddingModel(dimensions=16),
        )
    )

    assert backend.search("年假申请", document_ids=[], limit=1) == []


@pytest.mark.parametrize("limit", [0, -1])
def test_chroma_dense_backend_rejects_invalid_limit(tmp_path: Path, limit: int) -> None:
    backend = ChromaDenseSearchBackend(
        ChromaVectorStore(
            persist_directory=tmp_path,
            collection_name=f"invalid_limit_{abs(limit)}",
            embedding_model=HashEmbeddingModel(dimensions=16),
        )
    )

    with pytest.raises(ValueError, match="limit"):
        backend.search("年假申请", document_ids=None, limit=limit)


def test_chroma_dense_backend_rejects_empty_query(tmp_path: Path) -> None:
    backend = ChromaDenseSearchBackend(
        ChromaVectorStore(
            persist_directory=tmp_path,
            collection_name="empty_query_test",
            embedding_model=HashEmbeddingModel(dimensions=16),
        )
    )

    with pytest.raises(ValueError, match="query"):
        backend.search("  ", document_ids=None, limit=1)
