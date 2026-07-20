from pathlib import Path

from rag_agent_platform.models import ChildChunk, DocumentRecord, ParentChunk
from rag_agent_platform.storage.file_repository import FileDocumentRepository


def build_document(document_id: str = "doc-1") -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        filename="leave_policy.txt",
        file_type="txt",
        source_path="/tmp/leave_policy.txt",
        status="ready",
        metadata={"owner": "member1"},
    )


def build_parent(document_id: str = "doc-1") -> ParentChunk:
    return ParentChunk(
        chunk_id="parent-1",
        document_id=document_id,
        content="员工请假制度父块",
        metadata={"source": "leave_policy.txt"},
    )


def build_child(document_id: str = "doc-1", parent_id: str = "parent-1") -> ChildChunk:
    return ChildChunk(
        chunk_id="child-1",
        document_id=document_id,
        parent_id=parent_id,
        content="员工请假制度子块",
        metadata={"source": "leave_policy.txt", "parent_id": parent_id},
    )


def test_repository_saves_and_lists_document(tmp_path: Path) -> None:
    repository = FileDocumentRepository(tmp_path / "documents.json")
    document = build_document()

    repository.save_document(document)

    assert repository.get_document(document.document_id) == document
    assert repository.list_documents() == [document]


def test_repository_persists_data_across_instances(tmp_path: Path) -> None:
    storage_path = tmp_path / "documents.json"
    repository = FileDocumentRepository(storage_path)
    document = build_document()
    parent = build_parent()
    child = build_child()

    repository.save_document(document)
    repository.save_parent_chunks([parent])
    repository.save_child_chunks([child])

    reloaded = FileDocumentRepository(storage_path)

    assert reloaded.get_document(document.document_id) == document
    assert reloaded.get_parent_chunk(parent.chunk_id) == parent
    assert reloaded.list_child_chunks() == [child]


def test_repository_filters_child_chunks_by_document_ids(tmp_path: Path) -> None:
    repository = FileDocumentRepository(tmp_path / "documents.json")
    repository.save_child_chunks(
        [
            build_child(document_id="doc-1", parent_id="parent-1"),
            ChildChunk(
                chunk_id="child-2",
                document_id="doc-2",
                parent_id="parent-2",
                content="采购流程子块",
            ),
        ]
    )

    assert [chunk.document_id for chunk in repository.list_child_chunks(["doc-2"])] == ["doc-2"]
    assert repository.list_child_chunks([]) == []


def test_delete_document_removes_document_and_chunks(tmp_path: Path) -> None:
    repository = FileDocumentRepository(tmp_path / "documents.json")
    document = build_document()
    parent = build_parent()
    child = build_child()
    repository.save_document(document)
    repository.save_parent_chunks([parent])
    repository.save_child_chunks([child])

    repository.delete_document(document.document_id)

    assert repository.get_document(document.document_id) is None
    assert repository.list_parent_chunks() == []
    assert repository.list_child_chunks() == []
