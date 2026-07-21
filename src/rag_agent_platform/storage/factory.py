"""Factory for metadata repository implementations."""

from rag_agent_platform.config import Settings
from rag_agent_platform.storage.base import DocumentRepository
from rag_agent_platform.storage.file_repository import FileDocumentRepository
from rag_agent_platform.storage.mysql_repository import MySQLDocumentRepository


def build_document_repository(settings: Settings) -> DocumentRepository:
    """Build the configured document metadata repository."""
    provider = settings.document_repository_provider.strip().lower()
    if provider in {"", "file", "json"}:
        return FileDocumentRepository(settings.metadata_path)
    if provider == "mysql":
        return MySQLDocumentRepository(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password,
            database=settings.mysql_database,
            charset=settings.mysql_charset,
            connect_timeout=settings.mysql_connect_timeout,
        )
    raise ValueError(
        f"unsupported DOCUMENT_REPOSITORY_PROVIDER '{settings.document_repository_provider}'; "
        "use file or mysql"
    )
