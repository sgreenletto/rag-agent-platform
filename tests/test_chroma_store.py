from pathlib import Path

import chromadb
import pytest

from rag_agent_platform.embeddings import HashEmbeddingModel, OpenAICompatibleEmbeddingModel
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


def test_existing_collection_rejects_different_embedding_identity(tmp_path: Path) -> None:
    build_store(tmp_path, collection_name="identity_check")

    with pytest.raises(RuntimeError, match="Delete the old Chroma data"):
        ChromaVectorStore(
            persist_directory=tmp_path,
            collection_name="identity_check",
            embedding_model=HashEmbeddingModel(dimensions=32),
        )


def test_existing_collection_rejects_different_embedding_base_url(tmp_path: Path) -> None:
    ChromaVectorStore(
        persist_directory=tmp_path,
        collection_name="base_url_check",
        embedding_model=OpenAICompatibleEmbeddingModel(
            provider="openai_compatible",
            model="text-embedding-v4",
            api_key="test-key",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )

    with pytest.raises(RuntimeError, match="embedding_base_url"):
        ChromaVectorStore(
            persist_directory=tmp_path,
            collection_name="base_url_check",
            embedding_model=OpenAICompatibleEmbeddingModel(
                provider="openai_compatible",
                model="text-embedding-v4",
                api_key="test-key",
                base_url="https://example.com/v1",
            ),
        )


def test_non_empty_legacy_collection_without_embedding_metadata_is_rejected(
    tmp_path: Path,
) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection(
        name="legacy_without_identity",
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )
    collection.upsert(
        ids=["legacy-child-1"],
        documents=["旧向量库里已经存在的子块"],
        embeddings=[[0.1] * 16],
        metadatas=[
            {
                "chunk_id": "legacy-child-1",
                "document_id": "doc-1",
                "parent_id": "parent-1",
                "page": -1,
            }
        ],
    )

    with pytest.raises(RuntimeError, match="no embedding configuration metadata"):
        ChromaVectorStore(
            persist_directory=tmp_path,
            collection_name="legacy_without_identity",
            embedding_model=HashEmbeddingModel(dimensions=16),
        )


def test_empty_legacy_collection_without_embedding_metadata_can_be_initialized(
    tmp_path: Path,
) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.get_or_create_collection(
        name="empty_legacy_without_identity",
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )

    store = ChromaVectorStore(
        persist_directory=tmp_path,
        collection_name="empty_legacy_without_identity",
        embedding_model=HashEmbeddingModel(dimensions=16),
    )

    assert store.count() == 0
