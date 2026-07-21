"""MySQL-backed document repository for production metadata storage."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any, Protocol

import pymysql
from pymysql.cursors import DictCursor

from rag_agent_platform.models import ChildChunk, DocumentRecord, ParentChunk
from rag_agent_platform.storage.base import DocumentRepository


class CursorLike(Protocol):
    """Small DB-API cursor surface used by this repository."""

    def execute(self, query: str, args: object | None = None) -> int:
        """Execute one SQL statement."""
        ...

    def executemany(self, query: str, args: Iterable[object]) -> int:
        """Execute one SQL statement for multiple rows."""
        ...

    def fetchone(self) -> dict[str, Any] | None:
        """Fetch one row."""
        ...

    def fetchall(self) -> list[dict[str, Any]]:
        """Fetch all rows."""
        ...


class ConnectionLike(Protocol):
    """Small DB-API connection surface used by this repository."""

    def cursor(self) -> Any:
        """Return a cursor context manager."""
        ...

    def commit(self) -> None:
        """Commit the active transaction."""
        ...

    def rollback(self) -> None:
        """Roll back the active transaction."""
        ...

    def close(self) -> None:
        """Close the connection."""
        ...


class MySQLDocumentRepository(DocumentRepository):
    """Persist documents and chunks in MySQL tables."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 3306,
        user: str,
        password: str = "",
        database: str = "rag_agent",
        charset: str = "utf8mb4",
        connect_timeout: int = 5,
        initialize_schema: bool = True,
        connection_factory: Callable[[], ConnectionLike] | None = None,
    ) -> None:
        if not user.strip():
            raise ValueError("MYSQL_USER is required when DOCUMENT_REPOSITORY_PROVIDER=mysql")
        if not database.strip():
            raise ValueError("MYSQL_DATABASE is required when DOCUMENT_REPOSITORY_PROVIDER=mysql")
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database
        self._charset = charset
        self._connect_timeout = connect_timeout
        self._connection_factory = connection_factory
        if initialize_schema:
            self.initialize_schema()

    def initialize_schema(self) -> None:
        """Create required metadata tables when they do not already exist."""
        self._execute_transaction(
            lambda cursor: [
                cursor.execute(statement)
                for statement in (
                    self._documents_schema(),
                    self._parent_chunks_schema(),
                    self._child_chunks_schema(),
                )
            ]
        )

    def save_document(self, document: DocumentRecord) -> None:
        """Create or replace a document record."""
        sql = """
            INSERT INTO documents
                (document_id, filename, file_type, source_path, status, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                filename = VALUES(filename),
                file_type = VALUES(file_type),
                source_path = VALUES(source_path),
                status = VALUES(status),
                metadata = VALUES(metadata),
                updated_at = CURRENT_TIMESTAMP
        """
        params = (
            document.document_id,
            document.filename,
            document.file_type,
            document.source_path,
            document.status,
            self._to_json(document.metadata),
        )
        self._execute_transaction(lambda cursor: cursor.execute(sql, params))

    def get_document(self, document_id: str) -> DocumentRecord | None:
        """Return one document, or None when it does not exist."""
        row = self._fetch_one(
            """
            SELECT document_id, filename, file_type, source_path, status, metadata
            FROM documents
            WHERE document_id = %s
            """,
            (document_id,),
        )
        return None if row is None else self._document_from_row(row)

    def list_documents(self) -> list[DocumentRecord]:
        """Return all document records."""
        rows = self._fetch_all(
            """
            SELECT document_id, filename, file_type, source_path, status, metadata
            FROM documents
            ORDER BY filename, document_id
            """
        )
        return [self._document_from_row(row) for row in rows]

    def delete_document(self, document_id: str) -> None:
        """Delete a document and its stored chunks in one transaction."""

        def delete_all(cursor: CursorLike) -> None:
            cursor.execute("DELETE FROM child_chunks WHERE document_id = %s", (document_id,))
            cursor.execute("DELETE FROM parent_chunks WHERE document_id = %s", (document_id,))
            cursor.execute("DELETE FROM documents WHERE document_id = %s", (document_id,))

        self._execute_transaction(delete_all)

    def save_parent_chunks(self, chunks: list[ParentChunk]) -> None:
        """Persist parent chunks."""
        if not chunks:
            return
        sql = """
            INSERT INTO parent_chunks
                (chunk_id, document_id, content, page, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                document_id = VALUES(document_id),
                content = VALUES(content),
                page = VALUES(page),
                metadata = VALUES(metadata),
                updated_at = CURRENT_TIMESTAMP
        """
        params = [
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.content,
                chunk.page,
                self._to_json(chunk.metadata),
            )
            for chunk in chunks
        ]
        self._execute_transaction(lambda cursor: cursor.executemany(sql, params))

    def save_child_chunks(self, chunks: list[ChildChunk]) -> None:
        """Persist child chunks."""
        if not chunks:
            return
        sql = """
            INSERT INTO child_chunks
                (chunk_id, document_id, parent_id, content, page, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                document_id = VALUES(document_id),
                parent_id = VALUES(parent_id),
                content = VALUES(content),
                page = VALUES(page),
                metadata = VALUES(metadata),
                updated_at = CURRENT_TIMESTAMP
        """
        params = [
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.parent_id,
                chunk.content,
                chunk.page,
                self._to_json(chunk.metadata),
            )
            for chunk in chunks
        ]
        self._execute_transaction(lambda cursor: cursor.executemany(sql, params))

    def get_parent_chunk(self, chunk_id: str) -> ParentChunk | None:
        """Return one parent chunk for parent-context retrieval."""
        row = self._fetch_one(
            """
            SELECT chunk_id, document_id, content, page, metadata
            FROM parent_chunks
            WHERE chunk_id = %s
            """,
            (chunk_id,),
        )
        return None if row is None else self._parent_from_row(row)

    def list_parent_chunks(self, document_ids: list[str] | None = None) -> list[ParentChunk]:
        """Return parent chunks, optionally filtered by document IDs."""
        if document_ids == []:
            return []
        where, params = self._document_filter(document_ids)
        rows = self._fetch_all(
            f"""
            SELECT chunk_id, document_id, content, page, metadata
            FROM parent_chunks
            {where}
            ORDER BY document_id, chunk_id
            """,
            params,
        )
        return [self._parent_from_row(row) for row in rows]

    def list_child_chunks(self, document_ids: list[str] | None = None) -> list[ChildChunk]:
        """Return child chunks, optionally filtered by document IDs."""
        if document_ids == []:
            return []
        where, params = self._document_filter(document_ids)
        rows = self._fetch_all(
            f"""
            SELECT chunk_id, document_id, parent_id, content, page, metadata
            FROM child_chunks
            {where}
            ORDER BY document_id, chunk_id
            """,
            params,
        )
        return [self._child_from_row(row) for row in rows]

    def _connect(self) -> ConnectionLike:
        if self._connection_factory is not None:
            return self._connection_factory()
        try:
            return pymysql.connect(
                host=self._host,
                port=self._port,
                user=self._user,
                password=self._password,
                database=self._database,
                charset=self._charset,
                cursorclass=DictCursor,
                autocommit=False,
                connect_timeout=self._connect_timeout,
            )
        except pymysql.MySQLError as exc:
            raise RuntimeError(
                "Cannot connect to MySQL metadata database. "
                "Check MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD and MYSQL_DATABASE."
            ) from exc

    def _execute_transaction(self, operation: Callable[[CursorLike], object]) -> None:
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                operation(cursor)
            connection.commit()
        except pymysql.MySQLError as exc:
            connection.rollback()
            raise RuntimeError(f"MySQL metadata operation failed: {exc}") from exc
        finally:
            connection.close()

    def _fetch_one(self, sql: str, params: object | None = None) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql, params)
                return cursor.fetchone()
        except pymysql.MySQLError as exc:
            raise RuntimeError(f"MySQL metadata query failed: {exc}") from exc
        finally:
            connection.close()

    def _fetch_all(self, sql: str, params: object | None = None) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql, params)
                return list(cursor.fetchall())
        except pymysql.MySQLError as exc:
            raise RuntimeError(f"MySQL metadata query failed: {exc}") from exc
        finally:
            connection.close()

    @staticmethod
    def _document_filter(document_ids: list[str] | None) -> tuple[str, tuple[str, ...] | None]:
        if document_ids is None:
            return "", None
        placeholders = ", ".join(["%s"] * len(document_ids))
        return f"WHERE document_id IN ({placeholders})", tuple(document_ids)

    @staticmethod
    def _to_json(value: dict[str, Any]) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _from_json(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if value in (None, ""):
            return {}
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        parsed = json.loads(str(value))
        if not isinstance(parsed, dict):
            raise RuntimeError("metadata JSON must decode to an object")
        return parsed

    @classmethod
    def _document_from_row(cls, row: dict[str, Any]) -> DocumentRecord:
        return DocumentRecord(
            document_id=str(row["document_id"]),
            filename=str(row["filename"]),
            file_type=str(row["file_type"]),
            source_path=str(row["source_path"]),
            status=str(row["status"]),
            metadata=cls._from_json(row.get("metadata")),
        )

    @classmethod
    def _parent_from_row(cls, row: dict[str, Any]) -> ParentChunk:
        return ParentChunk(
            chunk_id=str(row["chunk_id"]),
            document_id=str(row["document_id"]),
            content=str(row["content"]),
            page=cls._normalize_page(row.get("page")),
            metadata=cls._from_json(row.get("metadata")),
        )

    @classmethod
    def _child_from_row(cls, row: dict[str, Any]) -> ChildChunk:
        return ChildChunk(
            chunk_id=str(row["chunk_id"]),
            document_id=str(row["document_id"]),
            parent_id=str(row["parent_id"]),
            content=str(row["content"]),
            page=cls._normalize_page(row.get("page")),
            metadata=cls._from_json(row.get("metadata")),
        )

    @staticmethod
    def _normalize_page(value: Any) -> int | None:
        if value in (None, ""):
            return None
        return int(value)

    @staticmethod
    def _documents_schema() -> str:
        return """
            CREATE TABLE IF NOT EXISTS documents (
                document_id VARCHAR(191) PRIMARY KEY,
                filename VARCHAR(512) NOT NULL,
                file_type VARCHAR(64) NOT NULL,
                source_path TEXT NOT NULL,
                status VARCHAR(64) NOT NULL,
                metadata JSON NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_documents_filename (filename),
                INDEX idx_documents_status (status)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """

    @staticmethod
    def _parent_chunks_schema() -> str:
        return """
            CREATE TABLE IF NOT EXISTS parent_chunks (
                chunk_id VARCHAR(191) PRIMARY KEY,
                document_id VARCHAR(191) NOT NULL,
                content LONGTEXT NOT NULL,
                page INT NULL,
                metadata JSON NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_parent_chunks_document_id (document_id),
                CONSTRAINT fk_parent_chunks_document
                    FOREIGN KEY (document_id)
                    REFERENCES documents(document_id)
                    ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """

    @staticmethod
    def _child_chunks_schema() -> str:
        return """
            CREATE TABLE IF NOT EXISTS child_chunks (
                chunk_id VARCHAR(191) PRIMARY KEY,
                document_id VARCHAR(191) NOT NULL,
                parent_id VARCHAR(191) NOT NULL,
                content LONGTEXT NOT NULL,
                page INT NULL,
                metadata JSON NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_child_chunks_document_id (document_id),
                INDEX idx_child_chunks_parent_id (parent_id),
                CONSTRAINT fk_child_chunks_document
                    FOREIGN KEY (document_id)
                    REFERENCES documents(document_id)
                    ON DELETE CASCADE,
                CONSTRAINT fk_child_chunks_parent
                    FOREIGN KEY (parent_id)
                    REFERENCES parent_chunks(chunk_id)
                    ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
