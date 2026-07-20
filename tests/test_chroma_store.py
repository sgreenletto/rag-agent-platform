from pathlib import Path

from rag_agent_platform.embeddings import HashEmbeddingModel
from rag_agent_platform.models import ChildChunk
from rag_agent_platform.storage.chroma_store import ChromaVectorStore


def build_store(tmp_path: Path, collection_name: str = "test_child_chunks") -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_directory=tmp_path,
        collection_name=collection_name,
        embedding_model=HashEmbeddingModel(dimensions=16),
    )


def build_child(
    chunk_id: str = "child-1",
    document_id: str = "doc-1",
    parent_id: str = "parent-1",
) -> ChildChunk:
    return ChildChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        parent_id=parent_id,
        content=f"{document_id} 的测试子块内容",
        metadata={
            "source": "leave_policy.txt",
            "file_type": "txt",
            "page": None,
        },
    )


def test_upsert_child_chunks_writes_vectors(tmp_path: Path) -> None:
    store = build_store(tmp_path)

    store.upsert_child_chunks([build_child()])

    assert store.count() == 1
    assert store.count("doc-1") == 1


def test_upsert_empty_chunks_is_noop(tmp_path: Path) -> None:
    store = build_store(tmp_path)

    store.upsert_child_chunks([])

    assert store.count() == 0


def test_delete_document_removes_only_matching_vectors(tmp_path: Path) -> None:
    store = build_store(tmp_path)
    store.upsert_child_chunks(
        [
            build_child(chunk_id="child-1", document_id="doc-1", parent_id="parent-1"),
            build_child(chunk_id="child-2", document_id="doc-2", parent_id="parent-2"),
        ]
    )

    store.delete_document("doc-1")

    assert store.count("doc-1") == 0
    assert store.count("doc-2") == 1
    assert store.count() == 1


def test_upsert_replaces_existing_chunk(tmp_path: Path) -> None:
    store = build_store(tmp_path)
    store.upsert_child_chunks([build_child(chunk_id="child-1")])
    store.upsert_child_chunks([build_child(chunk_id="child-1")])

    assert store.count() == 1
