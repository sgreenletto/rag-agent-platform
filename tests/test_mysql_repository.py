from collections.abc import Iterable
from typing import Any

import pytest

from rag_agent_platform.config import settings
from rag_agent_platform.models import ChildChunk, DocumentRecord, ParentChunk
from rag_agent_platform.storage.mysql_repository import MySQLDocumentRepository


class RecordingCursor:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.statements: list[tuple[str, object | None]] = []
        self.many_statements: list[tuple[str, list[object]]] = []
        self._rows = rows or []

    def __enter__(self) -> "RecordingCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: str, args: object | None = None) -> int:
        self.statements.append((" ".join(query.split()), args))
        return 1

    def executemany(self, query: str, args: Iterable[object]) -> int:
        args_list = list(args)
        self.many_statements.append((" ".join(query.split()), args_list))
        return len(args_list)

    def fetchone(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[dict[str, Any]]:
        return self._rows


class RecordingConnection:
    def __init__(self, cursor: RecordingCursor) -> None:
        self.cursor_obj = cursor
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self) -> RecordingCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


def build_repository(connection: RecordingConnection) -> MySQLDocumentRepository:
    return MySQLDocumentRepository(
        host="127.0.0.1",
        user="root",
        password="",
        database="rag_agent",
        initialize_schema=False,
        connection_factory=lambda: connection,
    )


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
        page=1,
        metadata={"source": "leave_policy.txt"},
    )


def build_child(document_id: str = "doc-1", parent_id: str = "parent-1") -> ChildChunk:
    return ChildChunk(
        chunk_id="child-1",
        document_id=document_id,
        parent_id=parent_id,
        content="员工请假制度子块",
        page=1,
        metadata={"source": "leave_policy.txt", "parent_id": parent_id},
    )


def test_mysql_repository_requires_user() -> None:
    with pytest.raises(ValueError, match="MYSQL_USER"):
        MySQLDocumentRepository(user="", database="rag_agent", initialize_schema=False)


def test_save_document_uses_upsert_and_commits() -> None:
    cursor = RecordingCursor()
    connection = RecordingConnection(cursor)
    repository = build_repository(connection)

    repository.save_document(build_document())

    assert "INSERT INTO documents" in cursor.statements[0][0]
    assert "ON DUPLICATE KEY UPDATE" in cursor.statements[0][0]
    assert connection.committed is True
    assert connection.closed is True


def test_delete_document_removes_chunks_then_document_in_one_transaction() -> None:
    cursor = RecordingCursor()
    connection = RecordingConnection(cursor)
    repository = build_repository(connection)

    repository.delete_document("doc-1")

    statements = [statement for statement, _params in cursor.statements]
    assert statements == [
        "DELETE FROM child_chunks WHERE document_id = %s",
        "DELETE FROM parent_chunks WHERE document_id = %s",
        "DELETE FROM documents WHERE document_id = %s",
    ]
    assert connection.committed is True


def test_row_conversion_restores_public_dataclasses() -> None:
    cursor = RecordingCursor(
        [
            {
                "document_id": "doc-1",
                "filename": "leave_policy.txt",
                "file_type": "txt",
                "source_path": "/tmp/leave_policy.txt",
                "status": "ready",
                "metadata": '{"owner":"member1"}',
            }
        ]
    )
    repository = build_repository(RecordingConnection(cursor))

    document = repository.get_document("doc-1")

    assert document == build_document()


def test_save_chunks_uses_batch_upsert() -> None:
    cursor = RecordingCursor()
    repository = build_repository(RecordingConnection(cursor))

    repository.save_parent_chunks([build_parent()])
    repository.save_child_chunks([build_child()])

    assert "INSERT INTO parent_chunks" in cursor.many_statements[0][0]
    assert "INSERT INTO child_chunks" in cursor.many_statements[1][0]
    assert cursor.many_statements[0][1][0][0] == "parent-1"
    assert cursor.many_statements[1][1][0][0] == "child-1"


def test_mysql_repository_contract_against_real_database() -> None:
    if settings.document_repository_provider.strip().lower() != "mysql":
        pytest.skip("set DOCUMENT_REPOSITORY_PROVIDER=mysql in .env to run real MySQL test")
    if not settings.mysql_user.strip():
        pytest.skip("set MYSQL_USER in .env to run real MySQL test")

    repository = MySQLDocumentRepository(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
        charset=settings.mysql_charset,
    )
    document = build_document("doc-mysql-contract")
    parent = build_parent(document.document_id)
    child = build_child(document.document_id, parent.chunk_id)

    repository.delete_document(document.document_id)
    repository.save_document(document)
    repository.save_parent_chunks([parent])
    repository.save_child_chunks([child])

    assert repository.get_document(document.document_id) == document
    assert repository.get_parent_chunk(parent.chunk_id) == parent
    assert repository.list_child_chunks([document.document_id]) == [child]

    repository.delete_document(document.document_id)

    assert repository.get_document(document.document_id) is None
    assert repository.list_child_chunks([document.document_id]) == []
